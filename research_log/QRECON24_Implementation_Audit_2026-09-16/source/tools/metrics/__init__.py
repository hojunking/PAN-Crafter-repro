"""PAN-Crafter 평가 지표 — DLPan-Toolbox 프로토콜의 파이썬 구현.

reduced-resolution : eval_rr.evaluate  (SAM / ERGAS / Q2n)
full-resolution    : eval_fr.load_dlpan, d_lambda_k, d_s  (D_lambda / D_s / HQNR)
Q2n 코어           : q2n.q2n

예전에는 옆 저장소(../CANConv/tools/)에서 import 했다. 그 파일들은 upstream
CANConv 에 없는 로컬 작성본이라, 다른 서버에서 clone 으로는 얻을 수 없었다.
이식성을 위해 이 저장소 안으로 옮겼다.

full-resolution 은 DLPan-Toolbox 의 wald_utilities.py (MTF / interp23tap) 를
런타임에 import 한다. DLPan-Toolbox 는 GPL-3.0 이라 이 저장소에 넣지 않고
외부 저장소로 두고 PANCRAFTER_DLPAN 환경변수로 위치를 알려준다.
"""
