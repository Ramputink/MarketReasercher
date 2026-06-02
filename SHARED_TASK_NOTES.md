# SHARED_TASK_NOTES — Loop autónomo 12h

> Ledger de continuidad entre ciclos del `/loop`. Cada ciclo LEE esto al despertar
> y AÑADE una entrada al final. Protocolo: docs/AUTONOMOUS_12H_RUNBOOK.md

```
loop_started_at: 2026-06-02 20:07:50 CEST
loop_deadline:   12h después de loop_started_at
mechanism:       /loop local self-paced
autonomy:        amplia, SIN live (cero órdenes/dinero)
branch:          auto/12h-loop
gate_command:    python tests/test_backtester_regression.py --check
benchmark:       python benchmark_evolution.py
evolution_log:   logs/evolve_12h_autonomous.log
evolution_state: reports/evolution_checkpoint.json
```

## Baselines al arrancar
- Velocidad (Fase 1): ~29.5 s/gen, ~122 gen/hora (9 workers, pop=40).
- Mejor estrategia conocida: volatility_squeeze (fitness 4.12, wf_sharpe 5.40).
- Golden harness: 12 casos verdes (tol 1e-9).

## Próxima idea (semilla para el ciclo 1)
- Track B: re-perfilar una strategy_fn y vectorizar su interior (siguiente cuello
  tras eliminar la materialización de filas pandas).
- Track A: deep-research de 2-3 indicadores/estrategias de tendencia-volatilidad no
  presentes aún en STRATEGY_REGISTRY; implementar la más prometedora.

---

## Bitácora de ciclos
(cada ciclo añade: timestamp · track · hipótesis · resultado/métricas · decisión · siguiente)

