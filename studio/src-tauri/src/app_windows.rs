//! Extra app windows. Each shows the web UI the backend serves, so it runs like a browser tab of
//! Unsloth Studio: no desktop IPC, the web browser panel, and a fresh owner session.

use std::sync::atomic::{AtomicU32, Ordering};

use log::warn;
use tauri::webview::NewWindowResponse;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_opener::OpenerExt;

use crate::diagnostics::DiagnosticsState;
use crate::process::BackendState;

pub const LABEL_PREFIX: &str = "app-window-";
static NEXT_WINDOW: AtomicU32 = AtomicU32::new(1);

pub fn origin(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// Runs before the page's scripts: marks the document as a web window (or the frontend would see
/// Tauri's injected globals and wait for desktop IPC), and seeds the session once per window.
pub fn init_script(origin: &str, tokens: Option<&(String, String)>) -> String {
    let origin = serde_json::to_string(origin).unwrap_or_else(|_| "\"\"".into());
    let tokens = match tokens {
        Some((access, refresh)) => {
            serde_json::to_string(&[access, refresh]).unwrap_or_else(|_| "null".into())
        }
        None => "null".into(),
    };
    format!(
        r#"(() => {{
  if (window.location.origin !== {origin}) return;
  Object.defineProperty(window, "__UNSLOTH_WEB_WINDOW__", {{ value: true }});
  const tokens = {tokens};
  if (!tokens || sessionStorage.getItem("unsloth_window_seeded")) return;
  sessionStorage.setItem("unsloth_window_seeded", "1");
  localStorage.setItem("unsloth_auth_token", tokens[0]);
  localStorage.setItem("unsloth_auth_refresh_token", tokens[1]);
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
    let (port, tokens) = crate::desktop_auth::window_session(&state, diagnostics.inner()).await?;
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
        .initialization_script(init_script(&origin, tokens.as_ref()))
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

    #[test]
    fn script_only_runs_on_the_backend_origin_and_escapes_tokens() {
        let script = init_script(&origin(8888), Some(&("a\"b".into(), "c</script>".into())));
        assert!(script.contains(r#"window.location.origin !== "http://127.0.0.1:8888""#));
        assert!(script.contains(r#"["a\"b","c</script>"]"#));
        assert!(script.contains("__UNSLOTH_WEB_WINDOW__"));
        assert!(init_script(&origin(1), None).contains("const tokens = null;"));
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
