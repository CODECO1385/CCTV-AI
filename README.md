# Flujo de personas — prototipo local (Streamlit + YOLO + ByteTrack)

Todo corre en tu equipo. Los videos solo se guardan en una carpeta temporal (`flujo_personas_*` dentro del directorio temporal del sistema) y se borran al pulsar «Iniciar análisis nuevo».

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
La primera ejecución descarga los pesos de YOLO (p. ej. `yolov8n.pt`); el video nunca se sube.
Sin GPU funciona en CPU: usa el modelo `n`, 640 px y 2–5 fps procesados.

## Ejecutar
```bash
streamlit run app.py
python -m pytest -q          # pruebas de los cálculos
```

## Uso
1. **Video**: sube MP4/MOV/AVI; verás duración, resolución y FPS.
2. **Espacio**: elige un fotograma, ajusta la línea (P1, P2 en % de la imagen) y el lado del *interior*; define zonas como polígonos (`x,y; x,y; …` en %) o con el rectángulo rápido. Puedes crear/editar/borrar filas. Guarda el JSON para reutilizarlo con la misma cámara (coordenadas normalizadas; avisa si la proporción cambia).
3. **Análisis**: modelo, confianza, fps, resolución. Usa el *modo de prueba* (20 s) primero.
4. **Resultados**: métricas, trayectorias, mapa de calor, flujo entre zonas, tabla por ID y descargas (CSV, JSON, video, ZIP).

## Definiciones y limitaciones
- **Tiempo observado**: primera a última detección de un ID. **Permanencia estimada**: cruce de entrada a cruce de salida del mismo ID; si falta alguno el registro es *incompleto*.
- Los IDs son temporales. Si ByteTrack pierde a alguien y reasigna otro ID, **no se unen**; solo se marcan *posibles cambios de ID*.
- Cruces con banda muerta de 10 px y cancelación de rebotes (<1.5 s). Transiciones entre zonas solo si hubo ≥0.5 s en cada zona.
- Las zonas se definen con coordenadas numéricas, no dibujando con el ratón (limitación de este MVP).
- El análisis es síncrono: la interfaz muestra progreso pero no se puede cancelar a mitad.
- El video procesado usa `mp4v`; algunos navegadores no lo reproducen (descárgalo).
- Video de prueba legal: p. ej. clips de Pexels o Pixabay (licencia libre) o MOT17/PETS09 (datasets de seguimiento).

## Sesiones, reporte, webhook y notificaciones (v1.1)
**Flujo:** *Iniciar sesión de análisis* (se ejecuta en segundo plano; puedes *Finalizar sesión ahora* y obtienes un reporte parcial marcado) → reporte estructurado → envío opcional.
**Reporte** (JSON/CSV, también dentro del ZIP): `session_id`, fechas UTC, duración analizada, personas únicas *estimadas* (IDs del rastreador; suele ser cota superior), detecciones totales, entradas/salidas (`null` sin línea), permanencia media (solo registros completos), actividad por intervalo, zonas, calidad y advertencias.

**Configuración (variables de entorno o *Secrets* de Streamlit; ver `.env.example`):** `WEBHOOK_URL`, `WEBHOOK_SECRET`, y para notificar `RESEND_API_KEY` o `SENDGRID_API_KEY` + `EMAIL_FROM`, o `TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN`/`TWILIO_FROM`. Los destinatarios se escriben en la pestaña *Integraciones* y no se guardan.

**Webhook:** POST JSON con cabeceras `X-Timestamp`, `X-Session-Id` y `X-Signature: sha256=HMAC(secreto, "<timestamp>." + cuerpo)`. Verifica la firma y rechaza timestamps viejos. Reintenta hasta 3 veces solo errores transitorios (red, 429, 5xx); no sigue redirecciones; exige HTTPS y bloquea hosts de red privada. Prueba rápida: crea un endpoint en webhook.site (o tu servidor), ponlo en `WEBHOOK_URL`, pulsa *Probar conexión* y luego analiza un video con envío automático.
**Datos:** el video y el video procesado viven en una carpeta temporal y se borran al iniciar un análisis nuevo; el reporte solo existe en la sesión del navegador hasta que lo descargues. El webhook envía solo estadísticas (nunca video ni imágenes); sin reconocimiento facial.
**Limitaciones:** no hay transmisión en vivo desde la cámara del navegador (requeriría `streamlit-webrtc` + HTTPS y, en la nube, un servidor TURN); las cámaras solo-Wi-Fi no se controlan desde el navegador: exporta/sube sus videos. Si el servidor se reinicia durante una sesión, esa sesión se pierde. «Aceptado por el proveedor» en SMS/correo no garantiza entrega.

## Cámara en vivo (v1.2)
Pestaña **🔴 Cámara en vivo**: el navegador envía la cámara (streamlit-webrtc, requiere HTTPS o localhost y permiso de cámara); un hilo aparte ejecuta YOLO+ByteTrack a un ritmo limitado (`intervalo mínimo`), así la imagen no se congela. En *Espacio* toma una foto de referencia y define línea/zonas. Al *Finalizar* se genera el reporte y los envíos con la misma lógica que el video cargado.
Límites: el ritmo de detección depende de la CPU del servidor (mídelo en tu equipo); en redes restrictivas la conexión WebRTC puede fallar sin servidor TURN; una cámara solo-Wi-Fi no es accesible desde el navegador. No se guarda video. Probado con pruebas automáticas de la lógica; **no** se probó en un navegador real con cámara.
