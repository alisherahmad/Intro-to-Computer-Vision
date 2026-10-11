"""Real-time HOG + SVM steel surface inspector (Lab 05 bonus). Usage: python realtime_inspector.py [source]"""
import sys, time, cv2, joblib, numpy as np
from skimage.feature import hog

cfg = joblib.load("steel_inspector.joblib")
P, S, TH, HP = cfg["patch"], cfg["stride"], cfg["threshold"], cfg["patch_hog"]

def feats(p):
    return hog(p, orientations=HP["orient"], pixels_per_cell=(HP["ppc"],) * 2,
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

def starts(n):
    s = list(range(0, n - P + 1, S))
    return s if s[-1] == n - P else s + [n - P]

src = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
while True:
    ok, frame = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    h, w = frame.shape[:2]; s = min(h, w)
    roi = cv2.cvtColor(frame[(h - s)//2:(h + s)//2, (w - s)//2:(w + s)//2], cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA)
    coords = [(y, x) for y in starts(200) for x in starts(200)]
    pr = cfg["patch_model"].predict_proba(np.stack([feats(roi[y:y+P, x:x+P]) for y, x in coords]))[:, 1]
    i = int(pr.argmax()); bad = pr[i] >= TH; conf = pr[i] if bad else 1 - pr[i]
    fps = 1 / (time.perf_counter() - t0)
    col = (40, 40, 220) if bad else (60, 180, 60)
    x0, y0 = (w - s) // 2, (h - s) // 2; k = s / 200
    cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (255, 255, 255), 1)
    if bad:
        y, x = coords[i]
        cv2.rectangle(frame, (x0 + int(x*k), y0 + int(y*k)), (x0 + int((x+P)*k), y0 + int((y+P)*k)), col, 2)
    cv2.rectangle(frame, (0, 0), (w, 50), col, -1)
    cv2.putText(frame, f"{'DEFECTIVE' if bad else 'PASS'}  {conf*100:.0f}%   {fps:.0f} FPS",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Steel surface inspection (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release(); cv2.destroyAllWindows()
