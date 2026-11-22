from overrides_helpers import *

def test_smoke(tmp_path):
    w = stopped_world(tmp_path)
    ids = attempts(w)
    print(ids)
    code, text = fork(w, ids[-1])
    print(code, text)
    assert code == 10
    print(cockpit_json(w, "cycles.json"))
    print((w.project/"carry"/"toy-2.json").read_text()[:600])
    assert any('"carry"' in p for p in w.claude.prompts)
    print(w.run("status")[1])
