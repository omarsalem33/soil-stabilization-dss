"""Thesis branding: academic credit header, logo slots, sidebar note and footer.

To show the official logos, save the image files in ``assets/logos/`` using these
base names (any of .png, .jpg, .jpeg, .webp or .svg is accepted):

    suez_canal_university.png      -> left logo slot
    faculty_of_engineering.png     -> right logo slot

Until the files exist, a neutral labelled placeholder is shown instead, so the
layout never breaks. Edit the constants below to change the credit text.
"""

import base64
import html
from pathlib import Path

import streamlit as st

LOGO_DIR = Path(__file__).resolve().parent.parent / "assets" / "logos"
LOGO_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".svg")
MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".svg": "image/svg+xml",
}

PROJECT_TITLE = "AI-Assisted Soil Stabilization Research DSS"
PROJECT_SUBTITLE = "Decision Support System for Stabilizing Expansive Soil with Lime, Rice Husk Ash and PET Fibers"
THESIS_LABEL = "Master's Thesis Research Project"
RESEARCHER = "Eng. Mohamed Hassan"
SUPERVISOR = "Prof. Dr. Waleed El-kamash and Prof. Dr. Azza Mubarak"
UNIVERSITY = "Suez Canal University"
FACULTY = "Faculty of Engineering"

# (file base name, placeholder caption, alt text)
LEFT_LOGO = ("suez_canal_university", "University logo", UNIVERSITY)
RIGHT_LOGO = ("faculty_of_engineering", "Faculty logo", f"{FACULTY}, {UNIVERSITY}")

_CSS = """
<style>
.thesis-header {
    background: linear-gradient(135deg, #0b2545 0%, #13315c 55%, #1d4e89 100%);
    color: #ffffff; border-radius: 14px; padding: 1.1rem 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 1.2rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(0,0,0,.18);
}
.thesis-header .logo-slot {
    flex: 0 0 110px; height: 110px; background: #ffffff; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; padding: 8px;
    box-sizing: border-box;
}
.thesis-header .logo-slot img { max-width: 100%; max-height: 100%; object-fit: contain; }
.thesis-header .logo-placeholder {
    width: 100%; height: 100%; border: 2px dashed #9aa7b8; border-radius: 8px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    color: #5b6b80; font-size: 0.72rem; text-align: center; line-height: 1.25; padding: 4px;
    box-sizing: border-box;
}
.thesis-header .thesis-text { flex: 1 1 auto; text-align: center; }
.thesis-header .thesis-badge {
    display: inline-block; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.35);
    border-radius: 999px; padding: 2px 12px; font-size: 0.78rem; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: 0.4rem;
}
.thesis-header h1 { color: #ffffff; margin: 0.1rem 0 0.2rem 0; padding: 0; font-size: 1.75rem; line-height: 1.2; }
.thesis-header .thesis-sub { margin: 0 0 0.55rem 0; font-size: 0.92rem; opacity: .9; }
.thesis-header .thesis-credit { margin: 0; font-size: 0.95rem; line-height: 1.55; }
.thesis-header .thesis-credit b { color: #ffd166; font-weight: 600; }
.thesis-footer { text-align: center; font-size: 0.82rem; opacity: .75; padding: 0.6rem 0 0.2rem 0; }
@media (max-width: 760px) {
    .thesis-header { flex-direction: column; }
    .thesis-header .logo-slot { flex-basis: 90px; width: 90px; height: 90px; }
    .thesis-header h1 { font-size: 1.35rem; }
}
</style>
"""


def _find_logo(base_name):
    for extension in LOGO_EXTENSIONS:
        path = LOGO_DIR / f"{base_name}{extension}"
        if path.is_file():
            return path
    return None


def _logo_slot(spec):
    base_name, caption, alt_text = spec
    path = _find_logo(base_name)
    if path is not None:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = MIME_TYPES[path.suffix.lower()]
        inner = f'<img src="data:{mime};base64,{encoded}" alt="{html.escape(alt_text)}">'
    else:
        inner = (
            f'<div class="logo-placeholder"><span>{html.escape(caption)}</span>'
            f'<span>assets/logos/<br>{html.escape(base_name)}.png</span></div>'
        )
    return f'<div class="logo-slot" title="{html.escape(alt_text)}">{inner}</div>'


def render_header():
    """Render the thesis banner (logos + title + researcher/supervisor/university)."""
    st.markdown(
        _CSS
        + f"""
<div class="thesis-header">
  {_logo_slot(LEFT_LOGO)}
  <div class="thesis-text">
    <span class="thesis-badge">{html.escape(THESIS_LABEL)}</span>
    <h1>{html.escape(PROJECT_TITLE)}</h1>
    <p class="thesis-sub">{html.escape(PROJECT_SUBTITLE)}</p>
    <p class="thesis-credit">
      Researcher: <b>{html.escape(RESEARCHER)}</b> &nbsp;|&nbsp;
      Supervisor: <b>{html.escape(SUPERVISOR)}</b><br>
      {html.escape(UNIVERSITY)} &mdash; {html.escape(FACULTY)}
    </p>
  </div>
  {_logo_slot(RIGHT_LOGO)}
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_credit():
    """Compact credit block at the top of the sidebar."""
    st.sidebar.markdown(
        f"**{THESIS_LABEL}**  \n"
        f"{RESEARCHER}  \n"
        f"Supervisor: {SUPERVISOR}  \n"
        f"{UNIVERSITY}, {FACULTY}"
    )
    st.sidebar.divider()


def render_footer():
    """Thesis credit line shown at the bottom of every page."""
    st.divider()
    st.markdown(
        f'<div class="thesis-footer">{html.escape(THESIS_LABEL)} &middot; '
        f"{html.escape(RESEARCHER)} &middot; Supervised by {html.escape(SUPERVISOR)} &middot; "
        f"{html.escape(UNIVERSITY)}, {html.escape(FACULTY)}</div>",
        unsafe_allow_html=True,
    )
