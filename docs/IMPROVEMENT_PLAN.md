# CryptoResearchLab — Plan de Mejora Masiva

> Estado vivo. Marcar `[x]` a medida que se completa. Última actualización: 2026-06.
>
> **Objetivo global:** (1) que el entreno corra a máxima velocidad en todos los cores del Mac
> (y con ayuda del GPU Metal), (2) que la estrategia sea *ultra-funcional* y estadísticamente
> honesta, y (3) lanzarlo en vivo con crons sin pausas y que genere rentabilidad real.

---

## Diagnóstico (estado a 2026-06)

Hallazgos verificados sobre el código actual:

| Capa | Estado | Evidencia |
|---|---|---|
| Evolución genética | ✅ Funciona | `auto_evolve.py:711` `ProcessPoolExecutor` |
| Paralelismo de entreno | ⚠️ Subóptimo | Pool recreado **cada generación** (`auto_evolve.py:720`) |
| Backtester (hot loop) | 🔴 Cuello #1 | Bucle Python `df.iloc[i]` bar a bar (`backtester.py:176`) |
| GPU Metal | 🔴 Apagada en entreno | Desactivada en workers (`auto_evolve.py:421-427`) |
| Resume/checkpoint | ⚠️ Solo escribe | `_save_checkpoint()` informativo (`auto_evolve.py:1064`) |
| Trading en vivo | 🔴 No existe | 0 órdenes, 0 scheduler |
| Consumo de best_genome | 🔴 Huérfano | Nada lee `reports/best_genome_*.json` para operar |

**Los 3 problemas que dominan el rendimiento de entreno:**

1. **Pool recreado cada generación** — con `spawn` (obligatorio en Mac) cada generación
   re-importa numpy/pandas/TF y recarga el DataFrame de disco. ~200-500ms × workers × 778 gen
   tirados. El pool debe vivir toda la corrida.
2. **Backtester Python puro fila a fila** — `df.iloc[i]` (acceso escalar pandas, lo más lento)
   y `strategy_fn(df, i, pos)` con el df entero, hasta 2x por barra. ~140k iteraciones Python
   por genoma. 10-50x más lento de lo posible.
3. **GPU Metal ocioso durante la evolución** — solo se usa para LSTM en el proceso principal.

---

## FASE 1 — Rendimiento y multicore (velocidad)

Meta: 5-10x más generaciones/hora saturando todos los P+E cores; mismo resultado científico.

- [x] **1.0 Benchmark base** — `benchmark_evolution.py`: mide s/gen, s/genoma, gen/hora offline.
- [x] **1.1 Pool persistente** — `ProcessPoolExecutor` creado una vez (`_get_executor`/`shutdown`,
  `auto_evolve.py`) + cache del DataFrame por worker (`_WORKER_DF_CACHE`): ~31k lecturas de disco → 1/worker.
- [x] **1.1b Backtester sin materializar filas pandas** — OHLC/timestamp/ATR a arrays numpy una
  vez; `_bar_regime` cacheado por df. Era el 70%+90% del coste (profiling). Bit-idéntico vs golden.
- [ ] **1.2 Memoria compartida del df** — `multiprocessing.shared_memory`/Arrow en vez de
  pickle a disco recargado por worker (gana RAM/arranque con muchos workers).
- [ ] **1.3 Vectorizar internos de estrategia / Numba** — siguiente cuello: cada `strategy_fn`
  aún usa pandas (`.iloc`, `.rolling`) por barra. (a) pre-computar señales vectorizadas por
  estrategia; (b) opcional kernel `@njit` de gestión de posición. **Nota:** el profiling mostró que
  el cuello era la materialización de filas pandas (ya resuelto), no la aritmética — por eso Numba
  se posterga hasta re-perfilar. Detrás de flag y validado contra el golden.
- [ ] **1.4 Folds walk-forward en paralelo** — solo si tras 1.3 sigue importando.
- [ ] **1.5 GPU Metal en evolución** — completar batch LSTM precompute; backtesting masivo GPU (I+D).

**Gate de Fase 1:** `tests/test_backtester_regression.py` (golden snapshot, tol=1e-9) garantiza
resultados idénticos ante cualquier optimización del backtester. **Ya en uso.**

### Progreso medido (2026-06)

| Versión | s/gen (steady) | gen/hora | genomas/hora |
|---|---|---|---|
| Pool recreado cada gen (original) | ~90 | ~40 | ~1.600 |
| + Pool persistente + cache df (1.1) | 60.8 | 59 | 2.370 |
| **+ Backtester/regime vectorizado (1.1b)** | **29.5** | **122** | **4.879** |

**~3x acumulado**, validado bit-idéntico (full-sample + walk-forward, 12 casos golden).
Medido en M-series 10 cores / 9 workers, pop=40, XRP 1h 365d.

---

## FASE 2 — Estrategia ultra-funcional (calidad de investigación)

Meta: que lo que evoluciona sea robusto de verdad, no overfit, con búsqueda más inteligente.

- [ ] **2.1 Resume real desde checkpoint** — persistir población + learning engine y reanudar.
  Crítico para crons. (`auto_evolve.py:1064`)
- [ ] **2.2 Island model** — N islas semi-independientes que migran los mejores cada K gen.
- [ ] **2.3 Validación anti-overfit** — purged/embargoed walk-forward, consistencia entre folds,
  penalización por complejidad y varianza.
- [ ] **2.4 Deflated Sharpe** — corregir por multiple-testing (31k+ genomas evaluados).
- [ ] **2.5 Hold-out temporal intocable** — últimos N meses reservados, reportar solo ahí.

---

## FASE 3 — Live + crons sin pausas (rentabilidad real)

Meta: cerrar research→live (hoy 0%). Siempre arrancar en paper/testnet.

- [ ] **3.1 `live_engine.py`** — websocket cierre de vela → cargar best_genome → strategy_fn → señal → SQLite.
- [ ] **3.2 Ejecutor de órdenes** — wrapper ccxt, sizing por volatilidad, slippage, reintentos. Paper primero.
- [ ] **3.3 Risk manager runtime** — DB de posiciones, kill-switch (pérdida diaria/drawdown/max pos), reconciliación.
- [ ] **3.4 Orquestación cron/systemd/launchd** — health-checks, restart on-failure, alertas, dashboard PnL.
- [ ] **3.5 Bucle research↔live cerrado** — re-evolución periódica que promociona a producción solo si pasa los gates de Fase 2.

---

## FASE 4 — Producción seria (transversal)

- [ ] Tests de regresión del backtester (Numba == Python).
- [ ] Reproducibilidad: seeds fijas, versionado de datos y genomas en producción.
- [ ] Observabilidad: latencia, fills, slippage real vs modelado.
- [ ] Gestión de secretos fuera del repo (`.env` ya ignorado).
- [ ] Modo degradado: si falla feed/modelo, cerrar y parar, no operar a ciegas.

---

## Orden de ejecución recomendado

| # | Acción | Fase | Esfuerzo | Impacto |
|---|---|---|---|---|
| 1 | Benchmark base | 1.0 | 🟢 | — (mide) |
| 2 | Pool persistente + df por worker | 1.1 | 🟢 | Alto |
| 3 | Resume real desde checkpoint | 2.1 | 🟢 | Alto (habilita cron) |
| 4 | Test de regresión backtester | 1.3/4.1 | 🟡 | Gate de seguridad |
| 5 | **Kernel Numba del backtester** | 1.3 | 🔴 | **Máximo** |
| 6 | Memoria compartida del df | 1.2 | 🟡 | Medio |
| 7 | Island model | 2.2 | 🟡 | Alto (calidad) |
| 8 | Gates anti-overfit + Deflated Sharpe + hold-out | 2.3-2.5 | 🟡 | Crítico |
| 9 | live_engine en paper/testnet | 3.1-3.3 | 🔴 | Habilita el objetivo |
| 10 | systemd/launchd + re-evolución cron | 3.4-3.5 | 🟡 | El "sin pausas" |

**Principio rector:** primero velocidad (desbloquea todo), luego honestidad estadística
(sin ella el live solo automatiza pérdidas), y solo entonces live (siempre en paper primero).
