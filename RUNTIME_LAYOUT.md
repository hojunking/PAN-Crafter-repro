<!-- S1_RUNTIME_RAID_RELOCATION_20260928 -->
> **s1 최신 상태 — 2026-09-28 RAID 이동 완료:** runtime 4개의 실제 파일은 `/raid/PAN_ARCHIVE/s1/runtimes/`에 있다. `~/Desktop/song/PAN-Crafter-runtime-*` 네 항목은 호환용 심볼릭 링크이며 로컬 실물 폴더가 아니다. 일반 파일 9,814개 SHA256, Git commit/수정 상태, Docker 읽기 검증 후 로컬 중복 사본을 제거했다. 데이터·중요 결과는 메인에 유지했다. 아래 기존 조사표보다 이 최신 이동 상태를 우선한다.

# PAN runtime 분류와 정리 기록

확인·처리: 2026-09-28. NAS는 모든 서버의 보관 선택지에서 제외한다. 데이터와 중요한 결과는 메인 레포의 현재 배치에 유지한다.

## 1. 전체 현황

runtime은 캠페인별 코드를 특정 commit에 고정한 Git worktree이다. Docker/Conda 환경 자체가 아니며 `.git` 객체 저장소를 메인과 공유한다. 데이터/결과 링크가 있는 runtime의 용량은 링크 대상 전체를 포함하지 않는다.

| 서버 | 조사 시 runtime | 작업 사본 정리 후 | 조사 시 용량 | 이번 정리 |
|---|---:|---:|---:|---:|
| s1 | 4 | 4 | 630.09 MiB | 0.00 MiB |
| s2 | 3 | 3 | 487.89 MiB | 0.00 MiB |
| s3 | 7 | 6 | 1058.02 MiB | 152.05 MiB |
| s4 | 10 | 9 | 1523.42 MiB | 152.05 MiB |
| s5 | 10 | 10 | 1523.22 MiB | 0.00 MiB |

34개를 조사했고 2개를 아카이브 보존 후 작업 사본만 제거했다. 나머지 32개는 현재 위치에 유지한다. main, release, ops worktree는 이번 삭제 대상이 아니다.
조사 시점에 runtime을 직접 사용하는 프로세스나 실행 중 컨테이너는 발견되지 않았다. 다만 s5의 별도 실행 중 컨테이너 `jingiju_govtech_yolo`가 `/home/user` 전체를 mount하므로 간접 접근 가능성은 남는다.

## 2. 분류와 실제 처리

- **수정본 보존 6개:** s1~s5의 G20 및 s1 QG40. 수정·미추적 파일을 포함하므로 단순 캐시로 지우지 않는다.
- **재현 경로 유지 25개:** 재개 receipt, 참조 모델 provenance 또는 종료된 컨테이너의 mount에서 사용한 경로이다. 경로 변경 시 이 연결을 같이 처리해야 한다.
- **s5 보류 1개:** GFB20 `3de787788c36`은 직접 참조가 발견되지 않았지만 위 실행 중 컨테이너의 상위 mount 범위에 들어간다.
- **정리 완료 2개:** s3·s4의 GFB20 `3de787788c36`. 수정·실행·컨테이너·실행 설정 참조가 없음을 재확인했다. 과거 캐시 정리 감사 기록의 언급은 의존 경로와 구분했다.

정리한 두 사본은 tar 내 일반 파일 2,226개씩의 SHA256을 현재 코드와 대조했다. tar에서 제외된 `pan_h5/pan_h5.zip`은 메인 원본과 동일한 SHA256임을 확인하고 메인 파일을 보존했다. 코드 commit도 기존 로컬 branch에서 계속 도달 가능하다.
`git worktree remove`를 사용했으며 강제 옵션은 쓰지 않았다. 공유 `data`, `work_dir`, 인증 파일 대상의 device/inode와 데이터 ZIP 해시가 유지됨을 사후 확인했다. 연구 결과의 내용은 삭제하지 않았다. 확보한 작업 사본 공간은 합계 약 304.10 MiB이다.

## 3. 권장 배치

현재 launcher는 runtime이 레포의 형제 디렉터리여야 한다는 조건을 검사한다(`ablr2/deployment.py::verify_release`). 단순히 레포 안으로 옮기면 그 검증과 Docker mount/재개 receipt가 달라진다. 따라서 다음처럼 **관리 기록은 레포 안에, 실행용 worktree는 현재 호환 위치에, 사용 종료 코드는 아카이브에** 둔다.

```text
PAN-Crafter/
  data/                       # 메인 데이터 유지
  work_dir/<중요 run>/         # 선택 모델·참조 모델과 결과 전체를 로컬 유지
  work_dir/<보관 run> -> <외장>/PAN_ARCHIVE/<server>/runs/...
  storage/ARCHIVE_INDEX.json   # 현재 보관 위치·UUID·검증 기록
  storage/RUNTIME_INDEX.json   # runtime별 용도·commit·수정 여부·분류
  RUNTIME_LAYOUT.md
PAN-Crafter-...-runtime-.../   # 현재 launcher가 기대하는 실행 경로
<외장>/PAN_ARCHIVE/<server>/source_snapshots/...tar
```

외장하드가 아직 없으므로 현재 별도 디스크/동일 디스크 staging을 유지한다. s3·s4는 외장 연결 전 보관 위치를 바꾸지 않는다. NAS를 대안으로 사용하지 않는다.
중요 결과는 선택 checkpoint 하나만 의미하지 않는다. 해당 run의 config·지표·선택 근거·후보 checkpoint·재개 상태·결과를 함께 로컬에 유지한다. 보존 대상 연구가 아직 확정되지 않은 경우 기존 PROTECT_STUDY도 계속 보존한다.
데이터가 들어 있는 `work_dir/.../data_view`와 참조 모델의 의존 파일도 같이 로컬에 유지한다. 외장으로 이관하는 것은 중요 결과로 지정되지 않은 보관용 run 전체와 사용 종료 코드 보관본이다.
원래 run 경로의 symlink와 레포 위치 색인을 함께 남기면 기존 코드가 경로를 계속 사용할 수 있다. Docker에는 링크가 가리키는 실제 외장 경로도 mount해야 한다. 외장이 분리된 경우 경로를 새 빈 폴더로 대체하지 말고 기록된 볼륨 UUID부터 확인한다.

## 4. 향후 runtime 정리 기준

- 실행·재개·참조 관계가 남은 사본은 경로를 유지한다. 이름이나 나이만으로 불필요 판정하지 않는다.
- 고유 수정 코드가 있는 사본은 실제 파일 snapshot, commit, 변경 목록, 공유 링크 목록을 보존한다.
- 참조가 없고 재생성 가능한 사본은 보관본의 파일 내용과 제외 데이터의 메인 원본을 검증한 뒤 Git worktree 등록과 작업 사본을 함께 정리한다.
- tar에 `.git`와 데이터가 포함된 것으로 가정하지 않는다. 복원 시 색인의 commit과 공유 데이터 경로를 함께 사용한다.
- 앞으로 실행 코드를 `runtimes/` 한 곳으로 모으려면 launcher, 경로 검증, Docker mount, 기존 receipt를 함께 바꾸는 별도 작업으로 진행한다. 이번에는 실험 코드를 변경하지 않았다.

## 5. 개별 runtime 목록

### s1

| runtime 이름 | commit | 처리 |
|---|---|---|
| `PAN-Crafter-runtime-ablr2-s1-6bb78e6d6118` | `6bb78e6d6118` | 재현 경로 유지 |
| `PAN-Crafter-runtime-ablr2x-s1-52184d5578a2` | `52184d5578a2` | 재현 경로 유지 |
| `PAN-Crafter-runtime-g20-s1-20260920` | `8a63d1060490` | 수정본 보존 |
| `PAN-Crafter-runtime-qg40-s1-20260920` | `cccedeeeffd7` | 수정본 보존 |

### s2

| runtime 이름 | commit | 처리 |
|---|---|---|
| `PAN-Crafter-gf2-release-runtime-ablr2-s2-1d1eb8abb6a3` | `1d1eb8abb6a3` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-ablr2x-s2-d035afa7f756` | `d035afa7f756` | 재현 경로 유지 |
| `PAN-Crafter-runtime-g20-s2-20260920` | `0ec113218f60` | 수정본 보존 |

### s3

| runtime 이름 | commit | 처리 |
|---|---|---|
| `PAN-Crafter-gf2-release-runtime-ablr2x-s3-c8fb7b21a212` | `c8fb7b21a212` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-gfb20-s3-060ba8d20992` | `060ba8d20992` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-gfb20-s3-3de787788c36` | `3de787788c36` | 아카이브 보존·작업 사본 제거 |
| `PAN-Crafter-gf2-release-runtime-gfp40-s3-0d2074f55f2e` | `0d2074f55f2e` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-l100-s3-bf9d6abab750` | `bf9d6abab750` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-pcrepro-s3-4e4bca82d235` | `4e4bca82d235` | 재현 경로 유지 |
| `PAN-Crafter-runtime-g20-s3-20260920` | `cccedeeeffd7` | 수정본 보존 |

### s4

| runtime 이름 | commit | 처리 |
|---|---|---|
| `PAN-Crafter-gf2-release-runtime-g23sens-s4-108bee06f373` | `108bee06f373` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-g23sens-s4-71e5faafed24` | `71e5faafed24` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-gfb20-s4-060ba8d20992` | `060ba8d20992` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-gfb20-s4-3de787788c36` | `3de787788c36` | 아카이브 보존·작업 사본 제거 |
| `PAN-Crafter-gf2-release-runtime-gfp40-s4-0d2074f55f2e` | `0d2074f55f2e` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-l100-s4-bf9d6abab750` | `bf9d6abab750` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-maina-hqnr-s4-454f66c3363b` | `454f66c3363b` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-maina-hqnr-s4-f1a97025088d` | `f1a97025088d` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-pcrepro-s4-4e4bca82d235` | `4e4bca82d235` | 재현 경로 유지 |
| `PAN-Crafter-runtime-g20-s4-20260920` | `cccedeeeffd7` | 수정본 보존 |

### s5

| runtime 이름 | commit | 처리 |
|---|---|---|
| `PAN-Crafter-gf2-release-runtime-g23sens-s5-03ecf56523f9` | `03ecf56523f9` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-g23sens-s5-0f06f9cdeb26` | `0f06f9cdeb26` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-g23sens-s5-24c64987f9ae` | `24c64987f9ae` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-gfb20-s5-060ba8d20992` | `060ba8d20992` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-gfb20-s5-3de787788c36` | `3de787788c36` | 다른 컨테이너 접근 가능: 보류 |
| `PAN-Crafter-gf2-release-runtime-gfp40-s5-0d2074f55f2e` | `0d2074f55f2e` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-l100-s5-bf9d6abab750` | `bf9d6abab750` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-maina-hqnr-s5-f1a97025088d` | `f1a97025088d` | 재현 경로 유지 |
| `PAN-Crafter-gf2-release-runtime-pcrepro-s5-4e4bca82d235` | `4e4bca82d235` | 재현 경로 유지 |
| `PAN-Crafter-runtime-g20-s5-20260920` | `cccedeeeffd7` | 수정본 보존 |

전체 절대 경로·tar SHA256·참조 파일·복원 데이터 경로는 [storage/RUNTIME_INDEX.json](storage/RUNTIME_INDEX.json)에 있다.
이번 갱신은 서버 working tree에 기록했으며 Git commit/push는 수행하지 않았다.

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
