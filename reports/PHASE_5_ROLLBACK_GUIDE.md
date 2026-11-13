# MINEGUARD AI — PHASE 5 ROLLBACK GUIDE

This document describes the exact procedure to safely rollback or disable the Phase 5 dashboard integration without data loss or corruption.

---

## 1. Safety Guarantees

- The original fallback repository (`D:\MINI_GUARD_PROJECT\MineGuard-AI`) was untouched throughout Phase 5 and remains an immutable reference.
- A pre-integration backup of `dashboard.py` was stored at:  
  `D:\Mineguard_project_2\backups\pre_phase5\dashboard.py.bak`
- A dedicated Git branch `phase5-dashboard-integration` was created for all Phase 5 changes.

---

## 2. Option A: Runtime Fallback (Zero Downtime)

If an operator or test engineer needs to switch the dashboard back to the original MineGuard core engine without stopping the server:

1. **Via UI:** Select `Original MineGuard Baseline (Fallback)` in the "Active Engine" dropdown.
2. **Via API:** Send a GET request:
   ```bash
   curl http://127.0.0.1:8080/api/set_engine?engine=MineGuardIntegratedSystem
   ```
The dashboard immediately releases DepthTrack models and routes all processing through `MineGuardIntegratedSystem`.

---

## 3. Option B: Complete Git Branch Rollback

To restore the repository state exactly to the Phase 4.5 audit commit (`dec7e16`):

```powershell
# 1. Switch back to main branch
git checkout main

# 2. Delete the integration branch if desired
git branch -D phase5-dashboard-integration

# 3. Verify clean state
git status
```

---

## 4. Option C: Restore Backed-Up File

To restore the pre-Phase 5 dashboard implementation manually:

```powershell
Copy-Item "D:\Mineguard_project_2\backups\pre_phase5\dashboard.py.bak" "D:\Mineguard_project_2\src\mineguard_depthtrack\dashboard.py" -Force
```

