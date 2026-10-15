# s1 WV3 / QB 실제 이미지 내보내기

지정된 TARGET 체크포인트(WV3 step 38380, QB step 47470)를 재추론하여 저장하는 폴더입니다.

- 각 센서 하위 폴더의 `Ours/<센서>/reduced/output/`, `full/output/`: 실제 예측의 PNG 미리보기.
- `Ours_raw/<센서>/reduced/pred/`, `full/pred/`: 전체 밴드 float32 DN 배열, 각각 20장.
- `references/`: 해당 테스트 장면의 원래 참조 배열.
- `metrics/`: 원래 기록과 저장 배열 재평가 및 차이.
- 완료된 센서는 ZIP 및 CRC/SHA256 검증 영수증이 생성됩니다. ZIP이 없으면 아직 처리 중입니다.

기존 학습을 중단하지 않기 위해 원래 Docker 이미지·코드를 사용하는 CPU 추론입니다. CUDA 원 실행과의 수치 차이는 별도 기록합니다. PNG는 원본 크기의 확인용 미리보기이며, CANConv 논문 그림의 renderer/ROI와 동일하다고 주장하지 않습니다.
