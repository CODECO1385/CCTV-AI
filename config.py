"""Lectura de secretos: variables de entorno primero, luego st.secrets. Nunca se imprimen ni se guardan."""
from __future__ import annotations
import os


def get_secret(name, default=None):
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        v = st.secrets.get(name)
        return v if v else default
    except Exception:
        return default
