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


### Ciclo 1 · 2026-06-02 20:37:23
- **Track:** diagnóstico + seed (Track A sin señal aún: gen 3, 0 robustos).
- **Estado evolución:** viva (PID 80429), gen ~3, re-buscando desde cero.
- **Hallazgo 1 (alto ROI):** el run NUEVO no reanuda — checkpoint marca gen 778 (Run #11) pero
  el proceso está en GEN 3. Cada run de 12h reinicia desde gen 0. → **Fase 2.1 (resume real)** es
  la mejora de mayor ROI. Campeones de Run #11 a salvo en git (cd3f2b4).
- **Hallazgo 2 (Track B):** perfilando volatility_squeeze, el 48% del tiempo sigue siendo
  `fast_xs` (df.iloc[i] DENTRO de la strategy_fn, ~5 materializaciones de fila/barra). Próximo
  cuello: vectorizar el acceso a datos dentro de las estrategias (numpy en vez de .iloc por barra).
- **Decisión:** Ciclo 2 = implementar la hipótesis de mayor ROI. Plan: empezar por Track B
  (vectorizar el interior de UNA estrategia, validar bit-idéntico contra golden) porque no
  depende de señal de evolución ni de benchmarking fiable bajo contención de CPU.
- **Gate:** no se tocó código → golden sigue verde por construcción.
- **Siguiente:** despertar en ~30 min; si hay robustos nuevos, evaluar pivote a Track A.

### Ciclo 2 · 2026-06-03 ~13:50 (objetivo: "10x en 6 meses, sin trampas")
- **Track:** A (objetivo de RETORNO, no solo Sharpe) + verificación adversarial (anti-trampa).
- **Veredicto honesto:** **10x en 6 meses NO es alcanzable honestamente** en estos datos
  (`any_honest_10x=false`, confirmado por 4 levers + 4 verificadores adversariales que reprodujeron
  y trataron de refutar cada número).
- **Baseline medido:** mejor genoma full-period real = donchian 1.045x/año; XRP buy&hold 0.56x
  (cayó 44%). Los retornos son minúsculos porque el motor optimiza Sharpe y despliega ~10% del capital.
- **Espejismo del 10x:** con leverage, vol_regime_arb da 10x@L5 / 30x@L8 *in-sample* — pero es trampa
  por (1) selección in-sample (params ajustados a esta serie; perturbar ±10% → 5.93x→~2.24x),
  (2) dependencia de una sola ventana (ninguna ventana independiente de 6m llega a 10x),
  (3) leverage sin modelo de liquidación: XRP tuvo una vela −49.75% → liquidación a **2x**.
- **Test decisivo (hold-out real):** entrenar en 60% / test en 40% no visto → vol_regime_arb ~**1.2x
  en ~5 meses** a L5 (DD 30%); volatility_squeeze = pérdida OOS (overfit). ~8–40x por debajo del objetivo.
- **Levers (todos fallan honestamente):** multi-activo HIERE (genoma XRP pierde en BTC/ETH/ADA);
  ensemble por régimen falla walk-forward (0.725x OOS); short = beta de caída, no alpha (se evapora en Q4).
- **Entregado:** `tools/honest_eval.py` (evaluador con modelo de liquidación + hold-out OOS real),
  `docs/HONEST_RETURNS_FEASIBILITY.md` (informe completo). Sin lookahead en el motor (verificado).
- **Substrato:** evolución 1.1h, 9 cores + GPU, 3.760 evals, 424 robustos; ningún campeón nuevo supera
  a los de git (los runs reinician desde gen 0 → **resume real sigue siendo la mejora de mayor ROI**).
- **Gate:** datos XRP se refrescaron (re-fetch Binance) y movieron la ventana 365d → golden quedó
  obsoleto; motor **bit-idéntico** (git diff vacío) → golden **regenerado**, verde 12/12 (tol 1e-9).
- **Siguiente:** reencuadrar objetivo a "máximo retorno OOS bajo presupuesto de DD"; arreglar resume;
  usar honest_eval como gate de reporte; pinear snapshot de datos para el golden.
