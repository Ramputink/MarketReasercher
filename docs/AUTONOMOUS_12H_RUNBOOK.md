# Runbook — Loop autónomo de 12 horas (búsqueda de mejores alternativas)

> Protocolo que sigue Claude en cada ciclo del `/loop` self-paced. Combina el
> **método** de ECC (`benchmark-optimization-loop`, `deep-research`, `eval-harness`)
> con el **mecanismo** de Claude Code (`/loop`) sobre el **sustrato** del lab
> (motor evolutivo en background, ya 3x más rápido tras Fase 1).
>
> Mecanismo: `/loop` local self-paced · Autonomía: amplia, SIN live · Stop: 12h.

---

## Objetivo (ambos tracks a la vez)

1. **Track A — Mejores estrategias:** descubrir/inventar/afinar estrategias más
   rentables y robustas (evolución continua + research de ideas nuevas).
2. **Track B — Motor más rápido:** continuar Fase 1 (vectorizar internos de
   estrategia, memoria compartida, GPU) para que la búsqueda explore más por hora.

Cada ciclo dedica esfuerzo a UN track (alternando), guiado por dónde hay más ROI
según el estado actual. El motor evolutivo corre en background todo el tiempo.

---

## Capa 1 — Sustrato (background, todos los cores + GPU)

- Un único proceso: `python auto_evolve.py --hours 12 --pop-size 40` corriendo en
  background = la búsqueda genética de estrategias.
- Log: `logs/evolve_12h_autonomous.log`. Estado: `reports/evolution_checkpoint.json`.
- **Regla de coordinación:** el loop NO lanza una segunda evolución. Solo la
  monitorea; si el proceso murió y quedan horas, la relanza una vez y lo anota.

---

## Protocolo por ciclo (self-paced, ~30–45 min)

Al despertar, en orden:

1. **Leer estado.** `SHARED_TASK_NOTES.md` (qué hizo el ciclo anterior), el tail de
   `logs/evolve_12h_autonomous.log`, y `reports/evolution_checkpoint.json`
   (generación, total_evaluated, total_robust, hall_of_fame, mejor fitness).
2. **Diagnosticar.** ¿Sube el mejor fitness robusto? ¿Qué estrategias dominan el
   hall of fame? ¿Se estancó (sin robustos nuevos en N gen)? ¿El motor sigue vivo?
3. **Elegir track** (A o B) según ROI. Aplicar el método ECC del track (abajo).
4. **Ejecutar UNA hipótesis** (variante única, medible).
5. **GATE de correctitud:** `python tests/test_backtester_regression.py --check`
   debe seguir verde tras cualquier cambio al backtester/estrategias.
6. **Aceptar o rechazar** según gates (abajo). Si se acepta y pasa, commitear en la
   rama `auto/12h-loop` (ver Autonomía).
7. **Registrar** en `SHARED_TASK_NOTES.md`: timestamp, track, hipótesis, resultado,
   métricas, decisión, y la próxima idea. Guardar aprendizajes en memoria.
8. **Re-agendar** el siguiente ciclo (ScheduleWakeup) salvo que se cumpla un stop.

### Track A — método (estrategias)
Sigue `.ecc/skills/benchmark-optimization-loop/SKILL.md` + `deep-research`:
- Baseline = mejor **wf_sharpe** robusto actual del hall of fame.
- Variantes (una hipótesis cada una): nueva estrategia desde literatura
  (`deep-research`/exa/web), nuevo espacio de parámetros, nuevo indicador, o nuevo
  término de fitness/gate anti-overfit.
- Implementar como módulo en `strategies/`, registrarlo en `STRATEGY_REGISTRY`,
  validar contra el golden, y dejar que la evolución en background lo recoja
  (o lanzar un mini-run dirigido para sembrarlo).
- Rechazar si no supera walk-forward robusto o empeora la robustez global.

### Track B — método (rendimiento)
Sigue el plan `docs/IMPROVEMENT_PLAN.md` (Fase 1.2/1.3) + `benchmark-optimization-loop`:
- Baseline = `python benchmark_evolution.py` (s/gen actual).
- Variantes: vectorizar internos de una `strategy_fn`, memoria compartida del df,
  batch LSTM en GPU.
- **Obligatorio:** cada cambio valida bit-idéntico contra `tests/test_backtester_regression.py --check`.
- Promover solo si es más rápido Y idéntico. Re-correr benchmark para confirmar delta.

---

## Gates (aceptar/rechazar)

- **Correctitud:** golden harness verde (`--check`, tol 1e-9). Innegociable.
- **Honestidad estadística:** una estrategia solo "cuenta" si es `wf_robust` en
  walk-forward; preferir las que además resistirían deflated-Sharpe / hold-out.
- **No regresión:** un cambio de rendimiento no puede alterar resultados; un cambio
  de estrategia no puede bajar el % robusto global del hall of fame.
- **Reproducibilidad:** seeds fijas; nada que dependa de azar no sembrado.

## Autonomía y seguridad (amplia, SIN live)

- ✅ Puede: investigar (web/exa/context7), implementar en working tree, crear la
  rama `auto/12h-loop`, commitear cambios que pasen los gates, abrir PRs, instalar
  dependencias que necesite (p.ej. `numba`).
- ⛔ Límite duro: **cero trading real, cero órdenes, cero claves de exchange, cero
  dinero**. Nada de `ccxt.create_order`, nada de testnet con fondos. Solo
  research + backtest.
- ⛔ No commitear secretos ni el log gigante. No tocar `main` directamente.
- Presupuesto: parar al llegar a 12h de reloj o si el coste supera el tope acordado.

## Condiciones de parada (terminar el loop, omitir ScheduleWakeup)

- Se cumplieron las **12 horas** desde el arranque (ver `loop_started_at` en notes).
- El usuario interviene.
- 3 ciclos seguidos sin progreso en ningún track (estancamiento) → resumir y parar.
- Un gate de correctitud no se puede recuperar (estado roto) → parar y reportar.

---

## El comando para arrancar (mecanismo)

```
/loop  Sigue docs/AUTONOMOUS_12H_RUNBOOK.md. Eres el operador del loop autónomo de
12h (Tracks A+B). Lee SHARED_TASK_NOTES.md, diagnostica el estado de la evolución,
ejecuta UNA hipótesis con el método ECC del track elegido, valida contra el golden
harness, registra el resultado y re-agenda el siguiente ciclo. Autonomía amplia
sin live. Para a las 12h o ante un stop del runbook.
```
(Self-paced: omite el intervalo; Claude marca su propio ritmo con ScheduleWakeup.)
