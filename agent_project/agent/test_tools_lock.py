"""Oturum 6 / madde 6: graph._tools() eşzamanlı çağrıda modelleri bir kez yüklemeli (gerçek model yok).

Kullanım: python -m agent.test_tools_lock
"""
import threading
import time
from collections import Counter

import agent.tools.hsemotion_tool as h
import agent.tools.mediapipe_tool as m
from agent import graph

created = Counter()


def fake(name):
    class Fake:
        def __init__(self):
            created[name] += 1
            time.sleep(0.5)  # yavaş model yüklemeyi taklit et -> yarış penceresi
    return Fake


def main():
    h.HSEmotionTool, m.MediaPipeTool = fake("hse"), fake("mp")
    graph._TOOLS.clear()
    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        results.append(graph._tools())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert created == Counter(hse=1, mp=1), created
    assert all(r is results[0] for r in results) and len(results) == 4
    print("OK: 4 eşzamanlı _tools() çağrısı -> her model 1 kez yüklendi", dict(created))


if __name__ == "__main__":
    main()
