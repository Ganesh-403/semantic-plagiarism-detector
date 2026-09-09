"""Show the deadline enforced by the server's session timeout check."""
import time
import streamlit.components.v1 as components


def render_session_countdown(last_interaction, timeout_seconds=900):
    deadline = int(((last_interaction or time.time()) + timeout_seconds) * 1000)
    components.html(f"""<div role="timer" aria-live="off" id="session-clock"></div>
<div role="alert" id="session-warning" hidden>Your session expires in less than two minutes. Use a page control to stay signed in.</div>
<script>
const deadline = {deadline};
function tick() {{
  const remaining = Math.max(0, Math.ceil((deadline-Date.now())/1000));
  document.getElementById('session-clock').textContent = 'Session remaining: ' + Math.floor(remaining/60) + ':' + String(remaining%60).padStart(2,'0');
  document.getElementById('session-warning').hidden = remaining > 120;
  if (remaining === 0) document.getElementById('session-warning').textContent = 'Session expired. Sign in again on your next interaction.';
}}
tick(); setInterval(tick, 1000);
</script>""", height=75)
