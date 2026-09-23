# PANDA G23 sensitivity / s4–s5

기준: WV3 · P0 · W104D121 · 고정 T0. 각 서버는 local BASE1 + variants6을 fresh50K로 수행한 뒤 새 seed로 반복한다. 시간/cycle/run 제한 없음. 서버 간 대기 없음.

1. `EXPERIMENT_PLAN_KR.md`: 전체 설계와14개 case.
2. `IMPLEMENTATION_HANDOFF_KR.md`: 실제 저장소/서버 runner 연결 계약.
3. `server_handoffs/s4.md`, `s5.md`: 서버별 전달문.
4. `case_templates.csv`, `examples/first_3_cycles/cases.csv`:14개 template와 최초42개 구체 run.
5. `tools/casegen.py`: 임의 cycle 생성과 lazy iterator. **학습 launcher는 아님.**
6. `config/runtime_bindings.example.json`: 로컬 확인 필요 값은 null. 임의 대체 금지.

검사:
```bash
python -m unittest discover -s tests -v
python tools/casegen.py emit --server both --start-cycle 0 --cycles 3 --out /tmp/g23_sens_preview
```

본 번들의 제공 상태는 case 설계와 생성기 구현까지다. GPU 학습/서버 전환/Sheet 쓰기는 이 대화에서 수행하지 않았다. 실제 연결 전 acceptance tests를 통과해야 한다.
