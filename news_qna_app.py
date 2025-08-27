# app.py
import os, re
from typing import List, Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="우리 연금술사", page_icon="📰", layout="centered")

# =========================
# ENV from st.secrets → os.environ
# =========================
def _prime_env_from_secrets():
    try:
        if hasattr(st, "secrets") and st.secrets:
            for k, v in st.secrets.items():
                os.environ.setdefault(k, str(v))
        else:
            st.warning("No secrets found in st.secrets. Ensure secrets are properly configured.")
    except FileNotFoundError:
        st.error("Secrets file not found. Please check your Streamlit configuration.")
    except Exception as e:
        st.error(f"Error loading secrets: {e}")

_prime_env_from_secrets()

# =========================
# 기본 유틸
# =========================
TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Seoul"))

def format_timestamp(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y년 %m월 %d일 %p %I:%M").replace("AM", "오전").replace("PM", "오후")

def _escape_html(s: Optional[str]) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _linkify(s: str) -> str:
    return re.sub(r"(https?://[\w\-\./%#\?=&:+,~]+)", r'<a href="\1" target="_blank">\1</a>', s or "")

def _build_messages_html(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for i, m in enumerate(messages):
        role = m.get("role", "assistant")
        row  = "user-row" if role == "user" else "bot-row"
        bub  = "user-bubble" if role == "user" else "bot-bubble"
        text_raw = m.get("content", "") or ""
        ts   = _escape_html(m.get("ts", ""))

        # 생성 중(typing) 말풍선
        if m.get("pending"):
            bubble = (
                '<div class="typing-bubble">'
                '<span class="typing-dot"></span>'
                '<span class="typing-dot"></span>'
                '<span class="typing-dot"></span>'
                '</div>'
            )
            parts.append(
                f'<div class="chat-row bot-row">{bubble}</div>'
                f'<div class="timestamp ts-left">{ts}</div>'
            )
            continue

        # 일반 버블
        text = _linkify(_escape_html(text_raw))
        parts.append(
            f'<div class="chat-row {row}"><div class="chat-bubble {bub}">{text}</div></div>'
            f'<div class="timestamp {"ts-right" if role=="user" else "ts-left"}">{ts}</div>'
        )

        # 소스칩 (assistant에만)
        if role == "assistant":
            srcs = m.get("sources") or []
            if srcs:
                chips = []
                for j, d in enumerate(srcs, 1):
                    md = (d.get("metadata") or {}) if isinstance(d, dict) else {}
                    title = md.get("title") or md.get("path") or md.get("source") or f"문서 {j}"
                    url = md.get("url")
                    try:
                        score = float(d.get("score", 0.0) or 0.0)
                    except Exception:
                        score = 0.0
                    label = f"#{j} {title} · {score:.3f}"
                    link_html = f'<a href="{url}" target="_blank">{label}</a>' if url else label
                    chips.append(f'<span class="source-chip">{link_html}</span>')
                parts.append(f'<div class="src-row">{"".join(chips)}</div>')

    # 본문 + 스크롤 앵커/스페이서
    return (
        '<div class="screen-shell">'
        '<div class="screen-body" id="screen-body">'
        + "".join(parts) +
        '<div class="screen-spacer"></div>'
        '<div id="end-anchor"></div>'
        '</div></div>'
        '<script>(function(){'
        ' document.addEventListener("click", function(ev){'
        '   var b = ev.target.closest(".copy-btn"); if(!b) return;'
        '   var txt = b.getAttribute("data-text") || "";'
        '   var ta = document.createElement("textarea"); ta.value = txt;'
        '   document.body.appendChild(ta); ta.select(); try{document.execCommand("copy");}catch(e){};'
        '   document.body.removeChild(ta);'
        ' }, true);'
        ' try {'
        '   var end = document.getElementById("end-anchor");'
        '   if (end) end.scrollIntoView({behavior:"instant", block:"end"});'
        ' } catch(e){}'
        '})();</script>'
    )

# =========================
# CSS
# =========================
st.markdown("""
<style>
:root{
  color-scheme: light !important;
  --brand:#0b62e6; --bezel:#0b0e17; --screen:#ffffff;
  --line:#e6ebf4; --chip:#eef4ff; --text:#1f2a44;
  --dock-h: 140px; /* 입력 Dock 전체 높이(그림자 포함) */
}
html, body, [data-testid="stAppViewContainer"]{ height: 100%; }
html, body, [data-testid="stAppViewContainer"], section.main, .stMain, [data-testid="stSidebar"]{
  background: radial-gradient(1200px 700px at 50% 0, #f0f4ff 0%, #f6f8fb 45%, #eef1f6 100%) !important;
  color: var(--text) !important;
}
.block-container > :first-child{
  position: relative !important;
  height: clamp(620px, 82vh, 860px);
  background: var(--screen) !important;
  border: 1px solid var(--line) !important;
  border-radius: 30px !important;
  padding: 12px 14px 14px !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.65);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.block-container > :first-child > div{
  display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0;
}
.screen-shell{ position: relative; display:flex; flex-direction:column; flex:1 1 auto; min-height:0; }
.block-container > :first-child .element-container:has(.screen-shell){ height: 100%; }
.screen-body{
  flex: 1 1 auto; min-height: 0; overflow-y: auto; touch-action: pan-y; -webkit-overflow-scrolling: touch;
  padding: 8px 10px 12px; scrollbar-width: thin; scrollbar-color: #c0c7d6 #f0f4ff;
}
.screen-body::-webkit-scrollbar{ width:8px; }
.screen-body::-webkit-scrollbar-track{ background:#f0f4ff; border-radius:8px; }
.screen-body::-webkit-scrollbar-thumb{ background:#c0c7d6; border-radius:8px; }
.screen-body::-webkit-scrollbar-thumb:hover{ background:#a0a7b6; }
.scree
