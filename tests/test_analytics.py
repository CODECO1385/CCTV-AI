import json
import numpy as np
import pandas as pd
import pytest
from analytics import analyze, frame_to_time, time_to_frame, line_events, zone_dwell, zone_transitions
from zones import SpaceConfig, EntryLine, Zone, parse_points, check_compatibility
import exports

SIZE = (1000, 500)


def cfg(zones=None):
    return SpaceConfig(EntryLine((0, 0.5), (1, 0.5), inside_side=1), zones or [], SIZE)


def df(rows):
    return pd.DataFrame(rows, columns=["id", "t", "x", "y"])


def path(pid, ts, ys, x=500):
    return [(pid, t, x, y) for t, y in zip(ts, ys)]


def test_frame_time():
    assert frame_to_time(30, 30) == 1.0 and frame_to_time(45, 15) == 3.0
    assert time_to_frame(2.5, 10) == 25
    with pytest.raises(ValueError):
        frame_to_time(1, 0)


def full_scenario():
    ts = list(range(20))
    ys1 = [100 + 30 * t for t in range(10)] + [370 - 30 * (t - 9) for t in range(10, 20)]
    return df(path(1, ts, ys1) + path(2, ts, [400] * 20))


def test_observed_and_complete_stay():
    r = analyze(full_scenario(), cfg(), SIZE, duration=20, sample_dt=1)
    p = r["people"].set_index("id")
    assert p.loc[1, "tiempo_observado_s"] == 19
    assert p.loc[1, "entrada_s"] == pytest.approx(5.0)
    assert p.loc[1, "salida_s"] == pytest.approx(13.0)
    assert p.loc[1, "permanencia_estimada_s"] == pytest.approx(8.0)
    assert p.loc[1, "estado"] == "Completo"
    assert r["summary"]["permanencia_media_s"] == pytest.approx(8.0)


def test_incomplete_record():
    r = analyze(full_scenario(), cfg(), SIZE, duration=20, sample_dt=1)
    p = r["people"].set_index("id")
    assert p.loc[2, "estado"].startswith("Incompleto")
    assert pd.isna(p.loc[2, "permanencia_estimada_s"])
    assert p.loc[2, "presente_al_inicio"] and p.loc[2, "presente_al_final"]
    assert r["summary"]["presentes_al_inicio"] == 1
    assert r["summary"]["registros_completos"] == 1


def test_unknown_is_none_not_zero():
    r = analyze(df(path(1, range(10), [100] * 10)), SpaceConfig(), SIZE, 10, 1)
    s = r["summary"]
    assert s["entradas_validas"] is None and s["presentes_al_inicio"] is None
    assert s["permanencia_media_s"] is None and s["zona_mayor_actividad"] is None


def test_no_duplicate_crossings_near_line():
    ys = [235, 265] * 10  # oscila fuera de la banda muerta
    ev = line_events([(t, 500, y) for t, y in enumerate(ys)], ((0, 250), (1000, 250)), 1, 10, 1.5)
    assert len(ev) <= 1
    ev2 = line_events([(t, 500, 250 + (-1) ** t * 5) for t in range(20)], ((0, 250), (1000, 250)), 1, 10, 1.5)
    assert ev2 == []  # dentro de la banda muerta: ningún cruce


def test_crossing_outside_segment_ignored():
    ev = line_events([(0, 1500, 200), (1, 1500, 300)], ((0, 250), (1000, 250)), 1)
    assert ev == []


def test_zone_time_and_transitions():
    zs = [Zone("A", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]), Zone("B", [(0.5, 0), (1, 0), (1, 1), (0.5, 1)])]
    t = df([(1, i, 50 + 100 * i, 100) for i in range(10)])
    r = analyze(t, cfg(zs), SIZE, 10, 1)
    zt = r["summary"]["tiempo_por_zona_s"]
    assert zt["A"] == 5 and zt["B"] == 4
    assert r["summary"]["transiciones"] == {"A → B": 1}
    assert r["summary"]["zona_mayor_actividad"] == "A"


def test_boundary_graze_not_counted():
    t = [0, 1, 2, 3, 4, 5, 6, 7]
    z = ["A", "A", "A", "B", "A", "A", "A", "A"]  # B solo 1 observación
    assert zone_transitions(t, z, min_dwell_s=0.5) == []
    assert zone_dwell(t, z, 3)["A"] > 0


def test_gap_not_counted_as_dwell():
    assert zone_dwell([0, 1, 20], ["A", "A", "A"], max_gap=3) == {"A": 1.0}


def test_possible_id_switch_flagged_not_merged():
    rows = path(1, range(0, 6), [100] * 6) + path(2, range(7, 13), [105] * 6)
    r = analyze(df(rows), SpaceConfig(), SIZE, 30, 1)
    assert len(r["people"]) == 2 and r["summary"]["posibles_cambios_id"]


def test_parse_points_and_compat():
    assert parse_points("0,0; 50,0; 50,100")[1] == (0.5, 0.0)
    for bad in ["0,0; 1,1", "a,b; 1,1; 2,2", "0,0; 150,0; 1,1"]:
        with pytest.raises(ValueError):
            parse_points(bad)
    assert check_compatibility(cfg(), (1000, 500)) is None
    assert check_compatibility(cfg(), (1080, 1920)) is not None


def test_config_roundtrip():
    c = cfg([Zone("Caja", [(0, 0), (1, 0), (1, 1)])])
    assert SpaceConfig.from_json(c.to_json()).to_dict() == c.to_dict()


def test_exports(tmp_path):
    r = analyze(full_scenario(), cfg(), SIZE, 20, 1)
    csv = exports.people_csv(r["people"])
    assert "Persona 1" in csv and "permanencia_estimada_s" in csv
    assert "entrada" in exports.events_csv(r["events"])
    j = json.loads(exports.summary_json(r["summary"], cfg()))
    assert j["resumen"]["ids_detectados"] == 2 and j["configuracion_espacio"]["line"]
    assert len(exports.build_zip(r["people"], r["events"], r["summary"], cfg())) > 100
