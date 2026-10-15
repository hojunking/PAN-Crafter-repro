<!-- S1_RUNTIME_RAID_RELOCATION_20260928 -->
> **s1 최신 상태 — 2026-09-28 RAID 이동 완료:** runtime 4개의 실제 파일은 `/raid/PAN_ARCHIVE/s1/runtimes/`에 있다. `~/Desktop/song/PAN-Crafter-runtime-*` 네 항목은 호환용 심볼릭 링크이며 로컬 실물 폴더가 아니다. 일반 파일 9,814개 SHA256, Git commit/수정 상태, Docker 읽기 검증 후 로컬 중복 사본을 제거했다. 데이터·중요 결과는 메인에 유지했다. 아래 기존 조사표보다 이 최신 이동 상태를 우선한다.

# PAN 보관 위치와 외장하드 우선 정책

갱신: 2026-10-07 (승인된 중복·과거 재개 이력 정리 반영; 기존 위치 이력 유지). 이 문서는 현재 실제 위치와 앞으로의 이동 원칙을 구분해 기록한다.
기존 `DISK_CLEANUP.md`의 일괄 checkpoint 삭제 지침보다 아래 사용자 보존 정책을 우선한다.

## 2026-10-07 s1 중복 정리·승인된 과거 재개 이력 삭제

사용자가 `과거 중간 재개 이력 삭제`를 명시 승인했다. 아래 범위는 기존의 모든 재개 이력 보존 원칙에 대한 **이 서버의 한정된 예외**다.

- 완료 B01/M12의 과거 중간 `training_state.pt` **1,248개 / 31.49 GiB**를 삭제했다. 파일별 SHA256을 사전 확인했으며 경로·해시·update·identity와 삭제 기록을 남겼다. 해당 과거 시점의 optimizer/RNG/sampler로 정확히 재개하는 기능은 복구할 수 없다.
- 초기(0), 최종(50000), 최신 `last` **24개**, 모든 공식 후보·선택 모델, 모든 평가 결과를 보존했다. `last` 해시 재검증과 대표 모델·재개 파일 Docker/CPU 읽기를 통과했다. 삭제한 디렉터리에는 `identity.json`과 `training_state.removed.json`이 남아 삭제 사유·최신 재개 경로를 확인할 수 있다.
- 불변 모델·예측의 바이트가 같은 사본 **398개 / 2.61 GiB**는 SHA256 대조 후 하드링크로 통합했다. 모든 파일 경로와 내용은 유지했다.
- 종료된 PAN 컨테이너 **23개**를 정리했다. 로그·실행 정보·고유 변경 파일을 보관했으며, 재생성 가능한 임시 캐시·드라이버 훅만 남은 컨테이너는 그 판정과 로그를 남겼다. 다른 프로젝트 컨테이너·공유 빌드 캐시·이미지는 정리 범위에 포함하지 않았다.
- 오래된 재생성 가능한 코드 캐시 정리: 125개. 이 작업의 대용량 파일 목록도 무손실 gzip으로 보관하고 복원 SHA256을 확인했다.
- 파일 내용 중복과 승인된 재개 이력 합계 **34.10 GiB**를 줄였다. 기록 시 메인 디스크 가용 공간은 **114.37 GiB**다. 실행 중인 TA2 컨테이너의 시작 시각은 유지됐고 재시작하지 않았다.
- **데이터/결과 이동 없음, 외부 아카이브 추가 사용 없음, NAS 사용 없음.**

실행 요약: [storage/2026-10-07_s1_cleanup.json](storage/2026-10-07_s1_cleanup.json). 상세 경로별 기록: `/home/knuvi/Desktop/song/PAN-Crafter/work_dir/_operator_storage/20261007-s1-cleanup`.

## 1. 사용자 결정

- 데이터와 중요한 결과는 **메인에 유지**한다. 선택·참조 모델의 run 전체와 미선별 보호 연구도 로컬 보존한다. 외장하드에는 그 외 보관용 완료 실험을 **run 전체 묶음**으로 옮기고 원래 경로에 심볼릭 링크와 위치 기록을 남긴다.
- HQNR 최고 모델만 남기지 않는다. ERGAS 등 다른 기준으로 선택한 모델, 모든 후보 checkpoint, optimizer/재개 상태, MAT 예측 결과를 사용자의 유의미한 실험 선별 전까지 보존한다.
- 보존은 위치를 고정한다는 뜻이 아니다. 의존 경로와 무결성을 확인한 뒤 전체 묶음의 보관 위치를 옮길 수 있다.
- **s3·s4는 외장하드가 연결될 때까지 현재 위치를 유지한다. NAS는 모든 서버에서 보관 선택지에서 제외한다.**
- 2026-09-28 확인 결과 USB/removable 외장하드는 발견되지 않았다. 아래 별도 SATA/NVMe 디스크를 USB 외장하드로 표현하지 않는다.
- runtime 34개를 점검하고 s3·s4의 미사용 GFB20 작업 사본 2개만 검증 후 제거했다. 코드 아카이브와 메인 데이터는 보존했다. [RUNTIME_LAYOUT.md](RUNTIME_LAYOUT.md)에 분류와 처리 기록이 있다.

## 2. 현재 실제 보관처

| 서버 | 현재 보관 루트 | 장치 성격 | 다음 단계 |
|---|---|---|---|
| s1 | `/raid/PAN_ARCHIVE/s1` | 레포 디스크와 별도 SATA 디스크 | 현재 보관 유지, 외장 연결 시 전체 아카이브 이전/복사 |
| s2 | `/mnt/nvme/archive/PAN_ARCHIVE/s2` | 별도 NVMe 파티션 | 현재 보관 유지, 외장 연결 시 이전/복사 |
| s3 | `/home/user/Desktop/song/PAN_ARCHIVE/s3` | 레포와 같은 디스크 | 외장 연결 전 현 위치 유지 |
| s4 | `/home/wty/Desktop/song/PAN_ARCHIVE/s4` | 레포와 같은 디스크 | 외장 연결 전 현 위치 유지 |
| s5 | `/home/user/AI_DATA/PAN_ARCHIVE/s5` | 별도 NVMe 디스크 | 현재 보관 유지, 외장 연결 시 이전/복사 |

s3·s4의 폴더 분리는 공간 확보나 독립 백업이 아니다. 모든 볼륨 UUID, mount 경로와 볼륨 기준 상대 경로는 [ARCHIVE_INDEX.json](storage/ARCHIVE_INDEX.json)에 있다.

기존 아카이브도 유지 중이다: s1 `/raid/pancrafter-archive-20260923`, s2 `/mnt/nvme/archive/pancrafter-20260923-s2`.
기존 아카이브는 당시 큰 파일만 옮긴 부분 보관일 수 있으므로, 레포의 잔여 metadata와 함께 취급한다. 이번에 전체 SHA 재검증한 것으로 간주하지 않는다.

## 3. 이미 실행한 이동과 보관

2026-09-28 1차 작업(`20260928-approved-01`)에서 완료 run 138개, 461.13 GiB를 아래처럼 재배치했다. 후보 모델과 실험 결과 내용은 버리지 않았다.

| 서버 | 이동 run | 전체 크기 | 처리 |
|---|---:|---:|---|
| s1 | 0 | — | core 4개 복사, 기존 아카이브 등록, 코드 snapshot |
| s2 | 0 | — | core 2개 복사, 기존 아카이브 등록, 코드 snapshot |
| s3 | 95 | 257.28 GiB | 같은 파일시스템 내 이동, 원래 경로 symlink |
| s4 | 29 | 160.51 GiB | 같은 파일시스템 내 이동, 원래 경로 symlink |
| s5 | 14 | 43.34 GiB | 별도 디스크 복사·모든 파일 SHA256 확인 후 원본 사본 해제, 원래 경로 symlink |

core 전체 복사본은 총 26개이고 원본은 유지했다. s3·s4 core 복사본도 같은 디스크에 있으므로 독립 백업은 아니다.
대표 safetensors/MAT 파일의 Docker/CPU 읽기를 확인했으며, GPU 추론이나 RR/FR 성능 재계산을 수행한 검증은 아니다.

실행 상세 기록:

- 레포 내부: `work_dir/_operator_storage/20260928-approved-01/result.json`, `postcheck.json`, `local_records/`.
- 보관처 내부: `<보관 루트>/manifest_index/20260928-approved-01/`의 파일별 SHA256 manifest와 실행 기록.
- 가장 최근 완료 기록: `<보관 루트>/manifest_index/latest-execution.json`.
- `storage-plan-20260928.json`은 실행 전 계획이다. 실제 이동 여부는 완료 실행 기록 및 이 위치 색인을 본다.

## 4. s1 레포 옆 runtime 디렉터리의 정체

네 디렉터리는 캠페인의 코드 버전을 고정한 **Git worktree**이다. 독립된 전체 Git 복제본이나 Python/Conda 환경이 아니다.
`ablr2/deployment.py`의 `git worktree add --detach`로 생성한 runtime은 메인 레포의 pull과 실험 코드를 분리하고, Docker가 그 경로의 코드를 사용하도록 한다.

| s1 디렉터리 이름 | 고정 commit | 사용 공간 | 관찰된 상태 |
|---|---|---:|---|
| `PAN-Crafter-runtime-ablr2-s1-6bb78e6d6118` | `6bb78e6d6118` | 151.25 MiB | 추적 코드 깨끗함, 공유 데이터/결과 symlink |
| `PAN-Crafter-runtime-ablr2x-s1-52184d5578a2` | `52184d5578a2` | 153.76 MiB | 추적 코드 깨끗함, 공유 데이터/결과 symlink |
| `PAN-Crafter-runtime-g20-s1-20260920` | `8a63d1060490` | 182.17 MiB | 추적 파일 수정 및 미추적 코드/기록 있음 |
| `PAN-Crafter-runtime-qg40-s1-20260920` | `cccedeeeffd7` | 142.91 MiB | 미추적 코드·config·연구 기록 있음 |

네 개 합계 약 **630.09 MiB**이다. ABLR2/R2X의 `data`와 `work_dir`는 메인 레포를 가리키므로 그 대용량 내용을 각각 복제한 것이 아니다.
별도의 `PAN-Crafter-gf2-release`, `PAN-Crafter-ops-qg40-s2-20260920`도 release/운영용 worktree이며, 두 개 합계 약 296.9 MiB이다.

네 runtime의 실제 코드 snapshot은 이미 다음 위치에 있다:

`/raid/PAN_ARCHIVE/s1/source_snapshots/20260928-approved-01/<runtime 디렉터리 이름>.tar`

이번 확인에서 tar SHA256과 현재 일반 파일 총 9,806개의 내용을 비교해 일치를 확인했다. 수정·미추적 코드도 snapshot에 포함되어 있다.
tar는 `.git`, 공유 데이터/결과, 인증 키를 제외한 코드 보관본이다. 연결된 연구 데이터나 Git 이력을 담은 독립 복제본은 아니다. 원래 worktree와 메인 Git 저장소는 유지 중이다.
worktree를 훗날 옮길 경우 컨테이너 bind 경로·symlink·Git 등록을 함께 갱신해야 하므로 일반 폴더처럼 무작정 `mv`/삭제하지 않는다.

## 5. 실험명으로 실제 위치 찾기

레포 루트에서 읽기 전용 조회:

```bash
python3 storage/find_archive.py PAKD50 --server s5
python3 storage/find_archive.py ablr2x --server s1
python3 storage/find_archive.py <run이름> --json
```

[STORAGE_LOCATIONS.md](STORAGE_LOCATIONS.md)는 요약, [storage/ARCHIVE_INDEX.json](storage/ARCHIVE_INDEX.json)은 전체 서버의 위치 색인이다.
색인은 원래 경로, 현재 보관 경로, 서버, 볼륨 UUID, 검증 수준, 파일 manifest 또는 tar SHA256을 담는다.
조회 시 현재 서버의 경로 존재 여부만 확인한다. 다른 서버를 SSH로 접속하거나 실험을 실행하지 않는다.
검색 결과가 없다는 것은 미등록일 수 있다는 뜻이며 삭제 가능 판정이 아니다. 저장 위치가 보이지 않을 때는 해당 볼륨의 연결·mount를 먼저 확인한다.

## 6. 외장하드 연결 후 적용할 순서

1. 장치 UUID와 실제 외장 연결을 확인하고 `<외장 mount>/PAN_ARCHIVE/<server>/`를 만든다. 용량은 현재 아카이브뿐 아니라 추가 보관할 완료 run과 여유 공간까지 포함해 산정한다.
2. 기존 `runs`, `core_backups`, `source_snapshots`, `manifest_index`와 구형 아카이브를 복사한다. core_backups의 외장 복사는 메인 중요 결과 원본을 대체하지 않는다. 현재 로컬에만 있는 완료 run 중 데이터·중요 결과·미선별 보호 연구를 제외한 보관용 run만 후보를 줄이지 않고 전체 묶음으로 추가한다. 실행 중이거나 의존 관계가 미확인인 것은 그 상태를 해소한 뒤 옮긴다.
3. 파일 목록·크기·SHA256, symlink 대상, 선택 checkpoint와 대표 MAT의 Docker 읽기를 확인한다. 원본 사본 해제는 검증된 항목에 한한다.
4. 원래 run 경로에는 위치 안내와 호환 링크를 남기고, 이 문서와 색인의 현재 경로/UUID/상대 경로/검증 시각을 갱신한다. 이전 경로도 이동 이력으로 유지한다. Git에 남길 것은 기록과 작은 manifest이며 모델/MAT/인증 키가 아니다.
5. Docker에서 `/workspace`만 mount하면 레포 밖 symlink가 끊길 수 있다. 사용 중인 실제 보관 루트도 **같은 절대 경로**로 bind mount하거나 경로 매핑을 명시한다. 서버별 현재 예시는 `work_dir/_operator_storage/20260928-approved-01/DOCKER_ARCHIVE_ACCESS.md`를 따른다.

실험은 계속 Docker로만 실행한다. 조회·파일 목록·무결성 점검은 실험 실행과 별개이다.
이번 기록 배치는 각 서버 working tree에 작성한 것이며 Git commit/push는 수행하지 않았다.

## 7. runtime 정리·데이터/중요 결과 로컬 보존

[RUNTIME_LAYOUT.md](RUNTIME_LAYOUT.md)와 [storage/RUNTIME_INDEX.json](storage/RUNTIME_INDEX.json)에 34개 runtime의 분류, 2개 작업 사본 제거 및 32개 유지 상태를 기록했다. NAS 제외와 데이터·중요 결과 메인 유지 원칙은 모든 서버에 적용한다.

## s1 runtime 실제 위치: 2026-09-28 최종 이동

| 기존 경로의 마지막 폴더명 — 현재는 링크 | 실제 경로 |
|---|---|
| `PAN-Crafter-runtime-ablr2-s1-6bb78e6d6118` | `/raid/PAN_ARCHIVE/s1/runtimes/PAN-Crafter-runtime-ablr2-s1-6bb78e6d6118` |
| `PAN-Crafter-runtime-ablr2x-s1-52184d5578a2` | `/raid/PAN_ARCHIVE/s1/runtimes/PAN-Crafter-runtime-ablr2x-s1-52184d5578a2` |
| `PAN-Crafter-runtime-g20-s1-20260920` | `/raid/PAN_ARCHIVE/s1/runtimes/PAN-Crafter-runtime-g20-s1-20260920` |
| `PAN-Crafter-runtime-qg40-s1-20260920` | `/raid/PAN_ARCHIVE/s1/runtimes/PAN-Crafter-runtime-qg40-s1-20260920` |

약 630.09 MiB의 runtime 실물이 RAID로 이동했다. `ls ../`에는 호환 링크의 이름이 보이며, `ls -ld ../PAN-Crafter-runtime-*`로 `-> /raid/...`를 확인할 수 있다. 이동 중 임시 로컬 사본도 제거 완료했다.

기존 Git worktree 등록 경로와 실험 receipt는 논리 경로를 유지한다. 해당 경로가 RAID를 가리키므로 Git 등록이나 불변 실험 기록의 내용을 바꾸지 않았다. 두 경로에서 Git commit과 수정 상태가 같음을 확인했다. G20/QG40의 수정·미추적 파일도 모두 보존했다.

`tools/ablr2_docker_start.py`는 기존 논리 경로와 RAID 실경로를 모두 Docker에 연결하도록 경로 처리만 보완했다. frozen runtime 코드 내용과 학습 방법은 바꾸지 않았다. CPU·읽기 전용 Docker 검증만 수행했으며 새 실험을 시작하지 않았다.

실행 기록: `work_dir/_operator_storage/20260928-s1-runtime-raid/result.json`.
RAID 기록: `/raid/PAN_ARCHIVE/s1/manifest_index/20260928-s1-runtime-raid/`.
`PAN-Crafter-gf2-release`, `PAN-Crafter-ops-qg40-s2-20260920`은 이번에 요청한 runtime 4개와 구분되는 release/ops 작업트리로, 이동 대상에 포함하지 않았다.


## 2026-10-13 사용 종료 형제 작업 사본 정리 완료

사용자 승인으로 PAN-Crafter-* 형제 작업 사본과 해당 Git worktree 등록을 정리했습니다. 메인 레포, 데이터, 실험 결과, 기존 코드 보관 tar는 유지했습니다. 개별 경로·commit·보관본·수정 파일·공유 링크 및 복원 정보는 `storage/20261013_sibling_worktree_cleanup/prepared.json`, 완료 기록은 같은 폴더의 `applied.json`에 있습니다. 각 commit은 `refs/archive/sibling-cleanup-20261013/`에 보존했습니다. 기존 실험을 재개하려면 기록된 원래 경로로 worktree와 코드 보관본/보충본, 공유 링크를 먼저 복원해야 합니다.
s1의 runtime 4개는 미마운트 RAID를 가리키던 링크만 제거했습니다. RAID 실물과 기존 보관 tar는 접근·삭제하지 않았습니다. RAID 재연결 시 Git worktree 등록 복구가 필요합니다.
