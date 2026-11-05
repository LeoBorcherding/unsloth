//! Extra app windows. Each shows the web UI the backend serves, so it runs like a browser tab of
//! Unsloth Studio: no desktop IPC, the web browser panel, and a fresh owner session.

use std::sync::atomic::{AtomicU32, Ordering};

use log::warn;
use tauri::webview::NewWindowResponse;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_opener::OpenerExt;

use crate::desktop_auth::WindowSession;
use crate::diagnostics::DiagnosticsState;
use crate::process::BackendState;

pub const LABEL_PREFIX: &str = "app-window-";
static NEXT_WINDOW: AtomicU32 = AtomicU32::new(1);

pub fn origin(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// Runs before the page's scripts: marks the document as a web window (or the frontend would see
/// Tauri's injected globals and wait for desktop IPC), and once per window stores the session, or
/// after a sign-out in the app drops the one an earlier window stored.
pub fn init_script(origin: &str, session: &WindowSession) -> String {
    let origin = serde_json::to_string(origin).unwrap_or_else(|_| "\"\"".into());
    let seed = match session {
        WindowSession::Tokens(access, refresh) => {
            serde_json::to_string(&[access, refresh]).unwrap_or_else(|_| "null".into())
        }
        WindowSession::SignedOut => "\"clear\"".into(),
        WindowSession::LoginRequired => "null".into(),
    };
    format!(
        r#"(() => {{
  if (window.location.origin !== {origin}) return;
  Object.defineProperty(window, "__UNSLOTH_WEB_WINDOW__", {{ value: true }});
  const seed = {seed};
  if (!seed || sessionStorage.getItem("unsloth_window_seeded")) return;
  sessionStorage.setItem("unsloth_window_seeded", "1");
  if (seed === "clear") {{
    for (const key of ["unsloth_auth_token", "unsloth_auth_refresh_token", "unsloth_auth_must_change_password", "unsloth_auth_session_mark"])
      localStorage.removeItem(key);
    return;
  }}
  localStorage.setItem("unsloth_auth_token", seed[0]);
  localStorage.setItem("unsloth_auth_refresh_token", seed[1]);
  if (!localStorage.getItem("unsloth_auth_session_mark"))
    localStorage.setItem("unsloth_auth_session_mark", `${{Date.now()}}.${{Math.random()}}`);
}})();"#
    )
}

pub fn same_origin(url: &tauri::Url, origin: &str) -> bool {
    url.origin().ascii_serialization() == origin
}

fn open_outside(app: &AppHandle, url: &tauri::Url) {
    if !matches!(url.scheme(), "http" | "https" | "mailto") {
        return;
    }
    if let Err(error) = app.opener().open_url(url.as_str(), None::<&str>) {
        warn!("Could not open {url} from an app window: {error}");
    }
}

pub async fn open(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<BackendState>();
    let diagnostics = app.state::<DiagnosticsState>();
    let (port, session, _signing_out_waits) =
        crate::desktop_auth::window_session(&state, diagnostics.inner()).await?;
    let origin = origin(port);
    let url: tauri::Url = format!("{origin}/chat")
        .parse()
        .map_err(|e| format!("{e}"))?;
    let label = format!(
        "{LABEL_PREFIX}{}",
        NEXT_WINDOW.fetch_add(1, Ordering::Relaxed)
    );
    let allowed = origin.clone();
    let (navigated, popped) = (app.clone(), app.clone());
    WebviewWindowBuilder::new(app, label, WebviewUrl::External(url))
        .title("Unsloth")
        .inner_size(1200.0, 800.0)
        .min_inner_size(480.0, 360.0)
        .initialization_script(init_script(&origin, &session))
        // Other sites, and every window.open, go to the default browser as a link from the app does.
        .on_navigation(move |url| {
            if same_origin(url, &allowed) {
                return true;
            }
            open_outside(&navigated, url);
            false
        })
        .on_new_window(move |url, _| {
            open_outside(&popped, &url);
            NewWindowResponse::Deny
        })
        .build()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Opens another app window on the running backend.
#[tauri::command]
pub async fn open_app_window(app: AppHandle) -> Result<(), String> {
    open(&app).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desktop_auth::DesktopAuthResponse;

    #[test]
    fn script_only_runs_on_the_backend_origin_and_escapes_tokens() {
        let session = WindowSession::Tokens("a\"b".into(), "c</script>".into());
        let script = init_script(&origin(8888), &session);
        assert!(script.contains(r#"window.location.origin !== "http://127.0.0.1:8888""#));
        assert!(script.contains(r#"const seed = ["a\"b","c</script>"];"#));
        assert!(script.contains("__UNSLOTH_WEB_WINDOW__"));
        assert!(
            init_script(&origin(1), &WindowSession::LoginRequired).contains("const seed = null;")
        );
    }

    #[test]
    fn a_sign_out_in_the_app_drops_what_an_earlier_window_stored() {
        let script = init_script(&origin(8888), &WindowSession::SignedOut);
        assert!(script.contains(r#"const seed = "clear";"#));
        for key in [
            "unsloth_auth_token",
            "unsloth_auth_refresh_token",
            "unsloth_auth_session_mark",
        ] {
            assert!(script.contains(&format!(r#""{key}""#)), "{key}");
        }
        let signed_out = DesktopAuthResponse::SignedOut {
            signed_out: crate::desktop_auth::LoginRequired,
        };
        assert_eq!(WindowSession::from(signed_out), WindowSession::SignedOut);
    }

    #[test]
    fn only_the_backend_origin_stays_in_the_window() {
        let origin = origin(8888);
        let ok = |url: &str| same_origin(&url.parse().unwrap(), &origin);
        assert!(ok("http://127.0.0.1:8888/chat?x=1"));
        assert!(!ok("http://127.0.0.1:8889/"));
        assert!(!ok("http://localhost:8888/"));
        assert!(!ok("https://127.0.0.1:8888/"));
        assert!(!ok("https://example.com/"));
    }
}
