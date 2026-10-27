"""Interfaz Streamlit: carga -> configuración -> análisis -> resultados."""
import json
import os
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import analytics as an
import camera
import exports as ex
import jobs
import notifier
import report as rp
import webhook as wh
import video_processor as vp
import visualizations as viz
import zones as zn

st.set_page_config(page_title="Flujo de personas", layout="wide")
ss = st.session_state
for k, v in {"tmpdir": None, "path": None, "info": None, "upload_key": None, "result": None,
             "zver": 0, "zones_df": pd.DataFrame({"Zona": pd.Series([], dtype="object"), "Puntos": pd.Series([], dtype="object")})}.items():
    ss.setdefault(k, v)


def cleanup():
    if ss.tmpdir and os.path.isdir(ss.tmpdir):
        shutil.rmtree(ss.tmpdir, ignore_errors=True)


def reset_all():
    j = ss.get("job")
    if j is not None:
        j.stop.set()
    cleanup()
    for k in list(ss.keys()):
        del ss[k]
    st.rerun()


def fmt(v, unit="s", nd=1):
    return "No se pudo determinar" if v is None else (f"{v:.{nd}f} {unit}".strip() if isinstance(v, float) else f"{v} {unit}".strip())


def s_(v):
    return "" if pd.isna(v) else str(v)


st.title("Análisis de movimiento y flujo de personas")
st.sidebar.markdown("**Privacidad**\n\nTodo se procesa en el equipo o servidor donde corre esta app (en la nube, el servidor del proveedor). Los videos se guardan solo temporalmente en:\n\n"
                    f"`{ss.tmpdir or tempfile.gettempdir() + '/flujo_personas_*'}`\n\n"
                    "y se borran al iniciar un análisis nuevo. Los IDs («Persona N») son temporales y no identifican a nadie.")
if st.sidebar.button("🔄 Iniciar análisis nuevo"):
    reset_all()

tab_live, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔴 Cámara en vivo", "Video (alternativa)", "Espacio", "Análisis de video", "Resultados", "Integraciones"])

# ---------------- 1. Video ----------------
with tab1:
    up = st.file_uploader("Carga un video (MP4, MOV o AVI)", type=["mp4", "mov", "avi"])
    if up is not None and ss.upload_key != (up.name, up.size):
        cleanup()
        ss.tmpdir = tempfile.mkdtemp(prefix="flujo_personas_")
        ss.path = os.path.join(ss.tmpdir, "video" + os.path.splitext(up.name)[1].lower())
        with open(ss.path, "wb") as f:
            f.write(up.getbuffer())
        ss.upload_key, ss.result = (up.name, up.size), None
        try:
            ss.info = vp.probe_video(ss.path)
        except vp.VideoError as e:
            ss.info = None
            st.error(str(e))
    if ss.info:
        i = ss.info
        c = st.columns(4)
        c[0].metric("Duración", f"{i.duration:.1f} s")
        c[1].metric("Resolución", f"{i.width}×{i.height}")
        c[2].metric("FPS", f"{i.fps:.1f}")
        c[3].metric("Fotogramas", i.frame_count)
        if i.duration > 240:
            st.warning("El video es largo (> 4 min). Puede tardar bastante; considera el modo de prueba.")
        if up is not None:
            st.video(up)
    st.caption("Cámara: en muchos celulares el selector de archivos permite grabar un video con la cámara y subirlo. "
               "La transmisión en vivo desde el navegador no está incluida en esta versión.")

# ---------------- 2. Espacio ----------------
cfg = zn.SpaceConfig()
with tab2:
    with st.expander("Modo cámara en vivo: toma una foto de referencia", expanded=not ss.info):
        snap = st.camera_input("Coloca la cámara como quedará durante el análisis y toma una foto", key="snap")
    ref = W = H = None
    if ss.info:
        W, H = ss.info.width, ss.info.height
        t_ref = st.slider("Fotograma de referencia del video (s)", 0.0, float(ss.info.duration), 0.0, 0.5)
        try:
            ref = vp.read_frame_at(ss.path, t_ref)
        except vp.VideoError as e:
            st.error(str(e)); st.stop()
    elif snap is not None:
        ref = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        H, W = ref.shape[:2]
    if ref is None:
        st.info("Toma una foto de referencia con la cámara o carga un video.")
    else:

        with st.expander("Cargar configuración guardada"):
            f = st.file_uploader("Archivo JSON de configuración", type=["json"], key="cfgup")
            if f is not None and st.button("Aplicar configuración"):
                try:
                    c = zn.SpaceConfig.from_json(f.getvalue().decode("utf-8"))
                    if c.line:
                        ss.use_line = True
                        ss.lx1, ss.ly1 = int(c.line.p1[0] * 100), int(c.line.p1[1] * 100)
                        ss.lx2, ss.ly2 = int(c.line.p2[0] * 100), int(c.line.p2[1] * 100)
                        ss.side = "Derecha" if c.line.inside_side == 1 else "Izquierda"
                    ss.zones_df = pd.DataFrame({"Zona": [z.name for z in c.zones],
                                                "Puntos": [zn.points_to_text(z.points) for z in c.zones]})
                    ss.zver += 1
                    w = zn.check_compatibility(c, (W, H))
                    ss.cfg_warning = w
                    st.rerun()
                except Exception as e:
                    st.error(f"Configuración inválida: {e}")
        if ss.get("cfg_warning"):
            st.warning(ss.cfg_warning)

        st.subheader("Línea de entrada/salida")
        use_line = st.checkbox("Usar línea de entrada/salida", value=True, key="use_line")
        c = st.columns(5)
        lx1 = c[0].slider("P1 x %", 0, 100, 10, key="lx1")
        ly1 = c[1].slider("P1 y %", 0, 100, 50, key="ly1")
        lx2 = c[2].slider("P2 x %", 0, 100, 90, key="lx2")
        ly2 = c[3].slider("P2 y %", 0, 100, 50, key="ly2")
        side = c[4].radio("Interior del local a la… (de P1 a P2)", ["Derecha", "Izquierda"], key="side")
        st.caption("La flecha amarilla «interior» indica el lado del local. Cruzar hacia ella = entrada; hacia el otro lado = salida.")

        st.subheader("Zonas de interés")
        with st.expander("Agregar rectángulo rápido"):
            q = st.columns(5)
            qn = q[0].text_input("Nombre", "Mesas")
            qx1, qy1 = q[1].number_input("x1 %", 0, 100, 0), q[2].number_input("y1 %", 0, 100, 0)
            qx2, qy2 = q[3].number_input("x2 %", 0, 100, 50), q[4].number_input("y2 %", 0, 100, 50)
            if st.button("Agregar zona"):
                txt = f"{qx1},{qy1}; {qx2},{qy1}; {qx2},{qy2}; {qx1},{qy2}"
                ss.zones_df = pd.concat([ss.zones_df, pd.DataFrame({"Zona": [qn], "Puntos": [txt]})], ignore_index=True)
                ss.zver += 1
                st.rerun()
        edited = st.data_editor(ss.zones_df, num_rows="dynamic", use_container_width=True, key=f"zed{ss.zver}",
                                column_config={"Puntos": st.column_config.TextColumn("Puntos (% x,y; x,y; …)")})
        zlist, errs = [], []
        for _, r in edited.iterrows():
            n, p = s_(r.get("Zona")).strip(), s_(r.get("Puntos")).strip()
            if not n and not p:
                continue
            try:
                zlist.append(zn.Zone(n or "Zona", zn.parse_points(p)))
            except ValueError as e:
                errs.append(f"{n or 'Zona'}: {e}")
        for e in errs:
            st.error(e)
        ss.zones_df = edited.reset_index(drop=True)
        line = zn.EntryLine((lx1 / 100, ly1 / 100), (lx2 / 100, ly2 / 100), 1 if side == "Derecha" else -1) if use_line else None
        cfg = zn.SpaceConfig(line, zlist, (W, H))
        img = ref.copy()
        viz.draw_space_overlay(img, cfg)
        st.image(img[:, :, ::-1], use_container_width=True)
        st.download_button("💾 Guardar configuración (JSON)", cfg.to_json(), "configuracion_espacio.json", "application/json")
        ss.cfg = cfg

# ---------------- 5. Integraciones (definida antes para que 3 lea sus valores) ----------------
with tab5:
    env = wh.env_settings()
    st.subheader("Webhook")
    st.checkbox("Enviar el reporte automáticamente al terminar la sesión", key="wh_auto")
    st.write(f"URL configurada en el servidor (`WEBHOOK_URL`): **{wh.mask_url(env['url'])}** · "
             f"secreto de firma (`WEBHOOK_SECRET`): **{'configurado' if env['secret'] else 'no configurado'}**")
    st.text_input("URL temporal solo para esta sesión (opcional, no se guarda)", type="password", key="wh_url_override",
                  help="Si la dejas vacía se usa WEBHOOK_URL del servidor. Debe ser HTTPS.")
    st.multiselect("Campos del reporte que se envían", rp.SELECTABLE_FIELDS, default=rp.SELECTABLE_FIELDS, key="wh_fields",
                   help="event, schema_version y session_id siempre se incluyen. Nunca se envían videos ni imágenes.")
    if st.button("Probar conexión"):
        e = wh.env_settings(ss.get("wh_url_override") or None)
        test = {"event": "people_flow.webhook.test", "schema_version": rp.SCHEMA_VERSION, "session_id": "test"}
        r = wh.send(e["url"], test, e["secret"], max_attempts=1, allow_private=e["allow_private"], allow_http=e["allow_http"])
        (st.success if r["ok"] else st.error)(
            f"Respuesta HTTP {r['status_code']}" if r["ok"] else f"Falló: {r['error']} (intentos: {r['attempts']})")
    st.subheader("Notificaciones (opcional)")
    av = notifier.available()
    c = st.columns(2)
    c[0].checkbox("Correo electrónico", key="em_on")
    c[0].text_input("Destinatario", key="em_to", placeholder="nombre@dominio.com")
    c[0].caption("Proveedor: " + (av["email"] or "no configurado (RESEND_API_KEY o SENDGRID_API_KEY + EMAIL_FROM)"))
    c[1].checkbox("SMS", key="sms_on")
    c[1].text_input("Teléfono (formato +521234567890)", key="sms_to")
    c[1].caption("Proveedor: " + (av["sms"] or "no configurado (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM)"))
    st.caption("Los destinatarios solo viven en esta sesión del navegador. Un mensaje se muestra como «aceptado» únicamente "
               "si el proveedor lo confirma; eso no garantiza la entrega.")

# ---------------- Cámara en vivo ----------------
with tab_live:
    if "live" not in ss:
        ss.live = camera.LiveSession()
    live = ss.live
    st.markdown("**Cómo usarlo:** 1) en *Espacio* toma una foto de referencia y define la línea/zonas (opcional); "
                "2) pulsa *Iniciar sesión* y permite el acceso a la cámara; 3) pulsa *Finalizar y generar reporte*.")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
    except ImportError:
        st.error("Falta la dependencia streamlit-webrtc (ver requirements.txt).")
        st.stop()
    c = st.columns(3)
    lmodel = c[0].selectbox("Modelo YOLO", ["yolov8n.pt", "yolo11n.pt", "yolov8s.pt"], key="lmodel", disabled=live.active)
    lconf = c[1].slider("Umbral de confianza", 0.1, 0.9, 0.35, 0.05, key="lconf", disabled=live.active)
    lwidth = c[2].select_slider("Resolución de detección (ancho)", [320, 480, 640, 960], 480, key="lwidth", disabled=live.active)
    linterval = st.slider("Intervalo mínimo entre detecciones (s)", 0.2, 2.0, 0.5, 0.1, key="linterval", disabled=live.active,
                          help="Más bajo = más preciso pero más carga de CPU. En servidores gratuitos usa 0.5–1.")
    b = st.columns(2)
    if b[0].button("▶️ Iniciar sesión", type="primary", disabled=live.active):
        def lfactory(m=lmodel, cf=lconf):
            from tracker import PersonTracker
            return PersonTracker(m, cf)
        ss.live_cfg = ss.get("cfg") or zn.SpaceConfig()
        ss.live_settings = jobs.SessionSettings(
            model=lmodel, conf=lconf, write_video=False, webhook_auto=bool(ss.get("wh_auto")),
            webhook_fields=list(ss.get("wh_fields") or rp.SELECTABLE_FIELDS), webhook_url=ss.get("wh_url_override") or None,
            email_to=(ss.get("em_to") or "").strip() or None if ss.get("em_on") else None,
            sms_to=(ss.get("sms_to") or "").strip() or None if ss.get("sms_on") else None)
        ss.live_session_id, ss.live_started = rp.new_session_id(), rp.utc_now()
        live.begin(ss.live_cfg, lfactory, lwidth, linterval)
        ss.result, ss.live_ended = None, False
    ctx = webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_frame_callback=live.callback,
                          rtc_configuration=camera.rtc_config(), media_stream_constraints={"video": True, "audio": False},
                          async_processing=True, desired_playing_state=True if live.active else (False if ss.get("live_ended") else None))

    def _live_status():
        if live.error:
            st.error(live.error)
        elif live.active:
            people = len({d[0] for d in live.dets})
            st.info(f"🔴 Sesión activa · modelo {'listo' if live.ready else 'cargando…'} · fotogramas procesados: "
                    f"{live.frames_processed} · personas en pantalla: {people}")
    st.fragment(run_every=1.0 if live.active else None)(_live_status)()

    if b[1].button("⏹ Finalizar y generar reporte", disabled=not live.active):
        try:
            tracks, pinfo, size, ref_frame = live.end()
            ss.result = jobs.finalize(tracks, pinfo, ss.live_cfg, size, ss.live_settings, ss.live_session_id,
                                      ss.live_started, ref_frame, source="camera_live")
            ss.live_msg = ("ok", "Sesión finalizada. Revisa «Resultados».")
        except vp.VideoError as e:
            ss.live_msg = ("error", str(e))
        ss.live_ended = True
        st.rerun()
    if ss.get("live_msg"):
        kind, text = ss.live_msg
        (st.success if kind == "ok" else st.error)(text)
    st.caption("El video de la cámara no se guarda. En la nube, el flujo pasa por el servidor de la app. "
               "Si la imagen no aparece, la red puede bloquear WebRTC: configura TURN_URL/TURN_USERNAME/TURN_CREDENTIAL.")

# ---------------- 3. Análisis ----------------
def _progress_panel():
    job = ss.get("job")
    if job is None:
        return
    if job.state == "running":
        st.progress(min(max(job.progress, 0.0), 1.0), text=job.message)
        if st.button("⏹ Finalizar sesión ahora", key="stopbtn"):
            job.stop.set()
            job.message = "Finalizando sesión…"
    elif not job.consumed:
        job.consumed = True
        if job.state == "done":
            ss.result = job.result
        st.rerun()
    if job.state == "error":
        st.error(job.error)
    elif job.state == "done":
        st.success("Sesión finalizada. Revisa «4. Resultados».")


with tab3:
    if not ss.info:
        st.info("Carga un video primero.")
    else:
        i = ss.info
        job = ss.get("job")
        running = bool(job and job.state == "running")
        c = st.columns(3)
        model = c[0].selectbox("Modelo YOLO", ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"], disabled=running,
                               help="El más pequeño (n) es el más rápido. Los pesos se descargan la primera vez.")
        conf = c[1].slider("Umbral de confianza", 0.1, 0.9, 0.35, 0.05, disabled=running)
        fps_t = c[2].select_slider("Fotogramas por segundo a procesar", [1, 2, 3, 5, 10, 15, 30], 5, disabled=running)
        c = st.columns(3)
        width = c[0].selectbox("Resolución de procesamiento (ancho)", [480, 640, 960, 1280, 0], 2, disabled=running,
                               format_func=lambda v: "Original" if v == 0 else f"{v}px")
        test_mode = c[1].checkbox("Modo de prueba (solo los primeros 20 s)", disabled=running)
        write_video = c[2].checkbox("Generar video procesado", True, disabled=running)
        if st.button("▶️ Iniciar sesión de análisis", type="primary", disabled=running):
            def factory(model=model, conf=conf):
                from tracker import PersonTracker
                return PersonTracker(model, conf)
            settings = jobs.SessionSettings(
                model=model, conf=conf, target_fps=float(fps_t), max_width=width, max_seconds=20 if test_mode else None,
                write_video=write_video, test_mode=test_mode, webhook_auto=bool(ss.get("wh_auto")),
                webhook_fields=list(ss.get("wh_fields") or rp.SELECTABLE_FIELDS), webhook_url=ss.get("wh_url_override") or None,
                email_to=(ss.get("em_to") or "").strip() or None if ss.get("em_on") else None,
                sms_to=(ss.get("sms_to") or "").strip() or None if ss.get("sms_on") else None)
            ss.job = jobs.Job(video_path=ss.path, tracker_factory=factory, cfg=ss.get("cfg") or zn.SpaceConfig(),
                              size=(i.width, i.height), settings=settings, session_id=rp.new_session_id(),
                              started_at=rp.utc_now(), out_video=os.path.join(ss.tmpdir, "procesado.mp4"))
            ss.result = None
            ss.job.start()
            st.rerun()
        st.fragment(run_every=1.0 if running else None)(_progress_panel)()

# ---------------- 4. Resultados ----------------
with tab4:
    R = ss.result
    if not R:
        st.info("Aún no hay resultados.")
    elif R["people"].empty:
        st.warning("No se detectaron personas (o los seguimientos fueron demasiado cortos). Prueba bajar el umbral de confianza o aumentar la resolución.")
    else:
        S, P, E = R["summary"], R["people"], R["events"]
        cols = st.columns(4)
        cols[0].metric("IDs detectados", S["ids_detectados"])
        cols[1].metric("Entradas válidas", fmt(S["entradas_validas"], ""))
        cols[2].metric("Salidas válidas", fmt(S["salidas_validas"], ""))
        cols[3].metric("Zona con más actividad", S["zona_mayor_actividad"] or "Datos insuficientes")
        cols = st.columns(4)
        cols[0].metric("Presentes al inicio", fmt(S["presentes_al_inicio"], ""))
        cols[1].metric("Presentes al final", fmt(S["presentes_al_final"], ""))
        cols[2].metric("Permanencia media / mediana", "Datos insuficientes" if S["permanencia_media_s"] is None
                       else f"{S['permanencia_media_s']:.1f} / {S['permanencia_mediana_s']:.1f} s ({S['registros_completos']} completos)")
        cols[3].metric("Tiempo observado promedio", fmt(S["tiempo_observado_medio_s"]))
        st.caption("«Tiempo observado» = primera a última detección. «Permanencia estimada» = cruce de entrada a cruce de salida del mismo ID. No son lo mismo.")

        st.subheader("Personas observadas en el tiempo")
        st.pyplot(viz.timeline_fig(R["timeline"]))

        st.subheader("Trayectorias")
        ids = sorted(P["id"].tolist())
        sel = st.multiselect("Trayectorias a mostrar", ids, default=ids[:10], format_func=lambda v: f"Persona {v}")
        st.image(viz.trajectories_image(R["ref"], R["tracks"], sel, R["cfg"]), use_container_width=True)

        st.subheader("Mapa de calor")
        mode = st.radio("Basado en", ["posiciones", "tiempo"], horizontal=True,
                        format_func=lambda m: "Concentración de posiciones" if m == "posiciones" else "Tiempo acumulado (s)")
        hm = viz.heatmap_image(R["ref"], R["tracks"], mode, R["max_gap"])
        if hm is not None:
            st.image(hm, use_container_width=True)

        st.subheader("Zonas y flujo")
        if S["tiempo_por_zona_s"]:
            st.dataframe(pd.DataFrame({"Zona": list(S["tiempo_por_zona_s"]), "Tiempo observado (s)": list(S["tiempo_por_zona_s"].values())}),
                         hide_index=True)
        else:
            st.write("Datos insuficientes: no hay zonas configuradas o nadie pasó por ellas.")
        if S["transiciones"]:
            st.pyplot(viz.flow_fig(S["transiciones"]))
            st.dataframe(pd.DataFrame({"Transición": list(S["transiciones"]), "Cantidad": list(S["transiciones"].values())}), hide_index=True)
        else:
            st.write("Transiciones entre zonas: datos insuficientes.")

        st.subheader("Detalle por identificador temporal")
        st.dataframe(P, use_container_width=True, hide_index=True)

        if R["video"]:
            st.subheader("Video procesado")
            st.video(R["video"])
            st.caption("Si no se reproduce en el navegador (códec mp4v), descárgalo y ábrelo con VLC.")
        elif R["pinfo"]["video_error"]:
            st.warning(R["pinfo"]["video_error"])

        st.subheader("Limitaciones y calidad")
        n_low = int((P["calidad"] == "Baja").sum())
        st.markdown(
            f"- Los IDs son temporales; si el rastreador pierde a alguien y lo recupera con otro ID, **no se unen**. "
            f"Posibles cambios de ID detectados: **{len(S['posibles_cambios_id'])}** (ver columna *notas*).\n"
            f"- Registros de calidad baja: **{n_low}** de {len(P)}. Los registros incompletos no representan la permanencia real.\n"
            f"- Se descartaron **{S['ids_descartados_por_ruido']}** seguimientos con muy pocas observaciones (probable ruido).\n"
            f"- Procesado a {R['pinfo']['fps_video'] / R['pinfo']['stride']:.1f} fps efectivos, {R['pinfo']['tamano_procesamiento'][0]}×{R['pinfo']['tamano_procesamiento'][1]} px; "
            f"errores del detector: {R['pinfo']['errores_detector']}.\n"
            "- Posición = centro inferior del cuadro; oclusiones y personas parcialmente visibles reducen la precisión.\n"
            "- Todas las cifras son estimaciones a partir de este video, no mediciones exactas.")

        if R.get("report"):
            rep = R["report"]
            st.subheader("Reporte de la sesión")
            st.caption(f"ID de sesión: {rep['session_id']} · inicio {rep['started_at']} · fin {rep['finished_at']} · "
                       f"calidad estimada: **{rep['quality']['level']}**")
            for w in rep["warnings"]:
                st.warning(w)
            with st.expander("Ver JSON del reporte"):
                st.json(rep)
            b = st.columns(2)
            b[0].download_button("Reporte JSON", json.dumps(rep, ensure_ascii=False, indent=2), "reporte.json", "application/json")
            b[1].download_button("Reporte CSV", rp.report_csv(rep), "reporte.csv", "text/csv")
            dl = R.get("deliveries") or []
            if dl:
                st.markdown("**Estado de envíos**")
                st.dataframe(pd.DataFrame(dl), hide_index=True, use_container_width=True)
            sent_ok = any(d["channel"] == "webhook" and d["ok"] for d in dl)
            if not sent_ok and wh.env_settings(ss.get("wh_url_override") or None)["url"]:
                if st.button("Enviar / reintentar webhook"):
                    payload, d = jobs.deliver_webhook(rep, list(ss.get("wh_fields") or rp.SELECTABLE_FIELDS),
                                                      ss.get("wh_url_override") or None)
                    R["deliveries"] = dl + [d]
                    R["webhook_payload"] = payload
                    st.rerun()

        st.subheader("Descargas")
        d = st.columns(5)
        d[0].download_button("CSV personas", ex.people_csv(P), "personas.csv", "text/csv")
        d[1].download_button("CSV eventos", ex.events_csv(E), "eventos.csv", "text/csv")
        d[2].download_button("JSON resumen", ex.summary_json(S, R["cfg"], R["settings"], R["pinfo"]), "resumen.json", "application/json")
        if R["video"]:
            with open(R["video"], "rb") as f:
                d[3].download_button("Video procesado", f.read(), "video_procesado.mp4", "video/mp4")
        d[4].download_button("Todo (ZIP)", ex.build_zip(P, E, S, R["cfg"], R["settings"], R["pinfo"], R["video"], R.get("report")), "resultados.zip", "application/zip")
        if st.button("🔄 Nuevo análisis"):
            reset_all()
