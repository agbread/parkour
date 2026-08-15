# 계단 보행 + Soft Dynamics 연구 브리핑 (새 세션용 컨텍스트)

> 이 문서는 다른 세션에 그대로 붙여넣어 **설명 없이 바로 토론을 이어가기 위한** 컨텍스트입니다.
> 핵심 주제: **"계단 보행(stair locomotion)에 soft dynamics 학습을 적용할 수 있는가"** + 관련 논문.

---

## 0. 프로젝트 배경 (한 줄 컨텍스트)

- **Robot Parkour Learning** (Zhuang et al., CoRL 2023) 코드베이스(`ZiwenZhuang/parkour`)를 **Unitree Go2**에 맞춰 적용 중.
- 목표 스킬: **run + stairsup + stairsdown** (논문 원래 5스킬 climb/leap/crawl/tilt/run과 다름).
- 방식: **combined-oracle** (`go2_field`가 여러 스킬 동시 학습) → distill.
- 파이프라인: flat walking 베이스(`go2_config`) → field oracle(`go2_field_config`, run+계단) → distill(`go2_distill`).
- walking 베이스는 flat(zScale=0.03)으로 재학습 + gait 보상(`feet_air_time`, `ang_vel_xy`, `orientation`, `lin_vel_z`, `action_rate`) 추가해 trot 안정화함.

## 1. 핵심 문제: 계단 앞에서 멈춤

- 첫 field run(stairsup/stairsdown만, hard, walking에서 resume) **실패 → 로봇이 계단 앞에서 멈춤.**
- 진단(지표): `terrain_level_stairs` ~0.3/2.0 정체, `n_obstacle_passed`<1, `timeout_ratio`~0.36, `tracking_lin_vel` 0.83→0.30, **`rew_lazy_stop` -0.09→-0.19 (멈춤 심화)**.
- 해석: **local optimum** — 계단 시도 후 넘어짐(termination, 큰 손해)보다 **멈춤(lazy_stop -3)이 덜 손해** → "멈추는 것"을 학습.

## 2. Soft Dynamics를 계단에 적용 — 핵심 토론 (이어갈 주제)

### 2-1. soft dynamics란 (논문)
- 장애물을 **penetrable(통과 가능)**하게 + **penetration-depth penalty** + **자동 커리큘럼**(soft→hard).
- 본래 **barrier(피해야 할 장벽)**용: climb/leap/crawl/tilt = "부피를 우회/극복".

### 2-2. 계단엔 soft가 코드상 막혀 있음
- `virtual_terrain=True`(soft) 시도 → **크래시**:
  `assert not virtual, "No virtual version of stairsup terrain"` (`barrier_track.py: get_stairsup_track`).
- 계단(stairsup/stairsdown)은 **soft 버전이 구현돼 있지 않음.**

### 2-3. 왜? (개념적 이유)
- **계단 = 지지면(support surface, 밟고 올라섬)**, barrier(피하는 것)가 아님.
- 통째로 penetrable로 만들면 → 로봇이 **밑 평지로 뚫고 떨어짐** → 계단 소멸(=평지) → 학습 신호 없음.
- barrier는 "침투=0 → 넘어감=정답"이지만, 계단은 발이 **표면에 닿아 지지받아야** 함. "침투 회피" ≠ "올라가기".

### 2-4. (반박 후 정밀화) naive soft만 불가, 계단 맞춤 soft는 원리상 가능
계단 한 칸 = **tread(수평 발판=지지)** + **riser(수직 면=회피 대상, 발끝이 stub되는 곳)**. 회피 요소는 riser.
- **"soft riser" 아이디어** (※ 내가 만든 표현, 논문 용어 아님): **riser만 penetrable, tread는 solid.**
  발이 riser를 뚫고 지나가되 tread에선 지지 → 전진 가능. riser 침투 penalty를 점점 키워 발을 들어 riser를 clear하게 학습. → "stub되어 멈춤" 실패를 직접 공략.
- **대안: ramp→stairs morph** 커리큘럼 (difficulty 0=경사로, 1=계단).
- **계단의 자연스러운 easing은 이미 존재 = height 커리큘럼**(낮은 단→높은 단). barrier의 penetration-easing에 대응하는 support-surface판 easing.
- → 결론: **둘 다 새 지형 코드 구현 필요** (코드에 없음). 비용 큼.

### 2-5. "계단 = jump의 연속?" (사용자 통찰)
- **운동 원형(motor primitive) 수준에선 맞음**: 각 단 = 작은 climb-up → **jump가 계단으로 전이되는 근거**.
- 하지만 **literal 반복 jump는 아님**: 연속 리듬/sequencing 필요, 다리가 동시에 여러 높이에 걸침, 에너지 효율, 그리고 **stairsdown=제어된 하강 ≠ jump**.
- 정리: **jump = "한 칸 올라타는 primitive" 제공 → 전이**, **계단 학습 = 그걸 연속 보행으로 잇는 sequencing 학습.**

## 3. 현재 채택한 접근 (진행 중)

- field options에 **jump + hurdle 재추가** (4스킬: jump, hurdle, stairsup, stairsdown), **hard**(virtual_terrain=False; 계단은 soft 불가).
- walking `model_8000`(안정 체크포인트)에서 resume.
- **가설**: jump/hurdle의 climb 동작이 계단으로 전이 → 멈춤 local optimum 탈출. (원래 성공한 10스킬 config도 hard였음 = 검증된 레시피.)
- **보류된 대안 (다음 카드)**: **jump/hurdle을 soft로 먼저 학습 → hard에서 계단 추가** (논문 충실, soft 지원되는 스킬에만 soft 적용). 전이가 부족하면 이걸로 escalate.

## 4. 찾은 논문 (웹 검증됨)

| 분류 | 논문 | 관련성 |
|---|---|---|
| soft dynamics 직계 | **Robot Parkour Learning** (Zhuang+, CoRL'23) | soft dynamics constraints 본체 |
| penetration 영감 | **Learning to Grasp the Ungraspable w/ Emergent Extrinsic Dexterity** (Wenxuan Zhou & David Held, CoRL'22) | parkour 논문이 명시한 "penetration 허용" 영감 (manipulation) |
| **가장 가까운 사촌** | **Contact-Implicit TO, Analytically Solvable Contact Model, Variable Ground** | 접촉(complementarity) 제약 **ε-relaxation→조이기**, 가변 지형 — "soft riser"에 개념적으로 최근접 |
| 접촉완화 기초 | **Contact-Implicit TO using Orthogonal Collocation** | contact 시퀀스 최적화 + 완화 |
| 제약 RL 대안 | **CaT: Constraints as Terminations** | 제약 위반을 확률적 termination으로; 실로봇 장애물 넘기 |
| 제약 RL parkour | **SoloParkour: Constrained RL for Visual Locomotion** | parkour를 제약 RL로 정식화 |
| 점프 커리큘럼 | **Curriculum-Based RL for Quadrupedal Jumping (reference-free)** | reference 없이 커리큘럼 점프 |

### 링크
- Robot Parkour Learning: https://proceedings.mlr.press/v229/zhuang23a/zhuang23a.pdf
- Emergent Extrinsic Dexterity: https://arxiv.org/abs/2211.01500 (code: https://github.com/Wenxuan-Zhou/ungraspable)
- Contact-Implicit TO (Orthogonal Collocation): https://ar5iv.labs.arxiv.org/html/1809.06436
- Contact-Implicit TO (Variable Ground, analytic contact): https://arxiv.org/pdf/2007.11261
- CaT: https://arxiv.org/abs/2403.18765 (project: https://constraints-as-terminations.github.io/)
- SoloParkour: https://arxiv.org/pdf/2409.13678
- Curriculum Quadrupedal Jumping: https://arxiv.org/pdf/2401.16337

## 5. 정직성 노트 (반드시 유지)

- **"soft riser"는 내가 만든 표현**, 그대로 하는 논문은 못 찾음 (추론).
- penetrable-stair-riser를 정확히 하는 선행연구 없음. **가장 가까운 정식 근거 = contact-implicit ε-relaxation**.

## 6. 새 세션에서 이어갈 열린 질문

1. **"soft riser" / ramp-morph**를 parkour 코드(`barrier_track.py`)에 실제 구현하는 방법? (per-face 충돌 속성, ramp 보간)
2. 논문의 **penetration-soft ↔ contact-implicit ε-relaxation**이 같은 아이디어의 다른 형식화인가?
3. **jump→계단 전이로 충분한가**, 아니면 직접 soft-stairs가 필요한가?
4. 커리큘럼이 **ramp→stairs / soft-riser→hard-riser**를 어떻게 연속 보간할까?
5. **CaT 방식**(제약=termination)으로 계단 "멈춤" 문제를 푸는 대안은?
