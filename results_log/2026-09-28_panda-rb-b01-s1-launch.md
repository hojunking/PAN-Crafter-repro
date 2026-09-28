# PANDA B01 s1 기동 기록

사용자 요청에 따라 2026-09-28 **19:21:19 KST**에 이 서버(s1)의 실험을 시작했다. s3/s5 기동과 git push는 수행하지 않았다.

- 명령: `python3 -B tools/panda_rb_start.py --server s1`
- Docker: `panda-rb-b01-s1`
- 최초 run: `RB01_WV3_S1_R1_SS9281101_QFULL_F50K_v1`
- 예산/범위: 등록 순서대로 Student 8개×50K, stress 16곡선 후 `STOP_FOR_REVIEW`. 새 seed/다음 batch 자동 추가 없음.
- 자동 preflight 통과: **19:22:40 KST**. 실제 batch48 CUDA parity, 원 Git reader parity, 공통 F1/6-seed map 검증, validation256 고정 F1 probe, 저장공간 검사를 통과했다.
- 실제 학습 진입: stdout에서 update100/200/300/400을 확인했다. update400의 `L_U=0.0464334`, `L_A=0.023009792`; 이는 초기 학습 loss이며 최종 성능 결과가 아니다.

## 실행 식별자

- image: `sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887`
- frozen release: `a0beb7e5711da8f86a29e184280edc4b453bedd1b0d7d99447c5909edaf67fa2`
- numerical source: `656b67d4c23bef830611db8676652a7760e88db3f2c2b3b17fec2f4f87abed34`
- common binding: `725958de631c87365ae1ac8f171909962600de751397775ed1e953e02293f72e`
- Teacher F1: `04ef8e756ae8b229c67ef14daf1912f0658f2e5c8bfe30afa9e0b476486fc519`
- QMEAN: `0.4996898875487226`

기동 시 Git HEAD는 `3094fcd2baf410f40d89f1ae67dbd44fcd17148d`였으며 새 구현은 아직 working tree에 있었다. 따라서 실행 코드는 HEAD만으로 식별하지 않고 위의 동결 소스/파일별 SHA manifest를 기준으로 한다. 이후 소스 커밋은 이미 실행 중인 동결 컨테이너의 수치 코드를 바꾸지 않는다.

## 확인/인계

```bash
docker logs -f panda-rb-b01-s1
python3 tools/panda_rb_runner.py status --server s1
```

상태·preflight·artifact 경로는 `work_dir/_panda_rb/20260928/B01/`이다. 단순 파일 존재만으로 완료를 판정하지 않고 runner가 SHA 및 선택점 identity를 재검증한다. 아직 50K 결과나 최종 native/stress 지표는 없다.

s1의 공통 e-cache와 여섯 mapping이 생성됐으므로, s3/s5에 전달할 package는 다음으로 만들 수 있다. 이 기록 작성 시 실제 export/전송/원격 기동은 하지 않았다.

```bash
python3 tools/panda_rb_transfer.py export work_dir/RB_B01_F1_package
```

커밋 범위는 B01 구현·계획·검증·평가 의존성과 B01 owner guard로 한정한다. 기존 R2 watchdog 변경, 다른 캠페인 변경, 연구 로그 이동/삭제, 기존 결과/Sheet는 포함하거나 수정하지 않는다.
