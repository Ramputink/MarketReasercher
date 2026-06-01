# CryptoResearchLab — Documentacion Completa del Sistema

## Plataforma Autonoma de Investigacion Cuantitativa y Trading Algoritmico

---

## 1. Vision General

CryptoResearchLab es un sistema de investigacion cuantitativa end-to-end disenado para descubrir, optimizar y validar estrategias de trading algoritmico en el par XRP/USDT de Binance. El sistema opera de forma autonoma: ingiere datos de mercado, genera hipotesis de trading, entrena modelos de machine learning, evoluciona parametros mediante algoritmos geneticos y valida todo mediante walk-forward testing estricto con proteccion contra overfitting.

El capital inicial objetivo es 200 USDC, con un enfoque en maximizar el ratio Sharpe ajustado por riesgo mientras se mantiene un drawdown maximo controlado del 15%.

### Principios de Diseno

El sistema se construye sobre cinco pilares fundamentales. Primero, cero fuga de informacion futura: cada indicador, cada prediccion y cada senal se computa usando exclusivamente datos pasados y presentes, nunca futuros. Segundo, validacion walk-forward obligatoria: ninguna estrategia se considera valida sin pasar por al menos 5 folds de train/validation/test temporal. Tercero, evolucion autonoma: el algoritmo genetico explora el espacio de parametros sin intervencion humana durante periodos de 8-14 horas. Cuarto, diversificacion de modelos: 7 variantes de clustering compiten entre si para que la evolucion descubra cual funciona mejor. Quinto, gestion de riesgo integrada: circuit breakers, position sizing por volatilidad y time stops protegen el capital en todo momento.

---

## 2. Arquitectura del Sistema

### Estructura de Directorios

El proyecto se organiza en modulos independientes que se comunican a traves de archivos de datos y configuraciones inmutables.

El directorio raiz contiene los scripts de orquestacion: `train.sh` es el punto de entrada maestro que gestiona todos los modos de ejecucion, `auto_evolve.py` implementa el motor genetico de 1620 lineas, `train_patterns.py` entrena los 14 modelos de clustering+LSTM, `run.py` ejecuta el pipeline completo de investigacion, y `config.py` define todas las configuraciones como dataclasses inmutables.

El directorio `engine/` contiene el nucleo computacional con aproximadamente 4230 lineas de codigo: el backtester event-driven, el sistema multi-clustering, el modelo LSTM, la ingenieria de features, las metricas de evaluacion, la ingestion de datos, el risk manager y las simulaciones Monte Carlo.

El directorio `strategies/` alberga 21 implementaciones de estrategias, de las cuales 11 estan activas en evolucion y 10 fueron podadas por rendimiento insuficiente.

Los directorios `data/`, `models/`, `checkpoints/`, `logs/` y `reports/` almacenan respectivamente los datos OHLCV en formato Parquet, los mas de 70 archivos de modelos entrenados, los checkpoints de TensorFlow, los logs de ejecucion y los reportes JSON de evolucion.

---

## 3. Configuracion del Sistema

### DataConfig — Pipeline de Datos

El sistema esta configurado para operar principalmente con el par XRP/USDT en Binance, con validacion cruzada contra BTC, ETH, SOL y ADA en USDT. Los timeframes soportados van desde 1 minuto hasta 1 dia (1m, 5m, 15m, 1h, 4h, 1d), siendo 1h y 4h los principales para evolucion. El historial por defecto es de 365 dias, y la validacion walk-forward utiliza ventanas de 90 dias para entrenamiento, 15 dias para validacion y 45 dias para test, distribuidos en 5 folds.

### BacktestConfig — Modelado de Costes

La simulacion incorpora costes realistas de ejecucion: comision del 0.1% por lado (taker fee de Binance), slippage promedio de 5 puntos basicos, posicion maxima del 10% del equity por trade, capital inicial simulado de 10,000 USD para backtesting (200 USDC para produccion), y periodo maximo de retencion de 48 horas.

Los criterios de aceptacion de una estrategia son: Sharpe ratio minimo de 1.0, drawdown maximo del 15%, minimo 30 trades en el periodo de test, profit factor superior a 1.0, y degradacion OOS (out-of-sample) menor al 50%.

### RiskConfig — Gestion de Posiciones

El risk manager opera con un maximo de 3 posiciones concurrentes, sizing por volatilidad apuntando a un 30% de vol anualizada, stop loss por defecto de 2x ATR, take profit de 3x ATR, time stop de 48 horas, intervalo minimo entre trades de 15 minutos, y un circuit breaker que detiene todo si el drawdown diario alcanza el 3% o el drawdown total el 10%.

### TFModelConfig — Optimizacion GPU Metal

El modelo LSTM principal (no el de patrones) utiliza secuencias de 60 barras, unidades LSTM de 64, capas densas de 128/64/32, batch size de 64 optimizado para M2 Metal, 100 epochs con early stopping a 15 de paciencia, y horizontes de prediccion de 1, 6, 12 y 24 horas con un umbral de etiqueta del 0.8% de movimiento.

---

## 4. Ingestion de Datos e Ingenieria de Features

### DataIngestionEngine

El modulo `engine/data_ingestion.py` (324 lineas) conecta con multiples exchanges via la libreria CCXT. Soporta Binance como fuente primaria, con Kraken y Coinbase como fuentes de validacion cruzada. Los datos se cachean en formato Parquet para evitar re-descargas, y el sistema gestiona automaticamente la actualizacion cuando los datos son mas antiguos que el historial configurado.

### Feature Engineering — 35 Indicadores sin Fuga

El modulo `engine/features.py` (348 lineas) computa 35 features estrictamente backward-looking. Cada feature usa unicamente datos pasados y presentes, garantizando cero contaminacion temporal.

Los indicadores tecnicos incluyen SMA y EMA en periodos de 20, 50 y 100 barras, RSI de 14 periodos, ADX de 14 periodos, ATR de 14 periodos, Bollinger Bands (20 periodos, 2 desviaciones) con su bandwidth y %B, niveles de retraccion Fibonacci (0.236, 0.382, 0.5, 0.618, 0.786), Bull Market Support Band (BMSB: SMA-20 vs EMA-21), MACD (12/26/9), RSI Estocastico, Williams %R y CCI.

Las metricas de volatilidad y microestructura comprenden volatilidad historica en ventanas de 10, 20 y 60 periodos, volatilidad Garman-Klass (estimador basado en OHLC), volatilidad Parkinson, ratio de volumen vs media de 20 barras, desequilibrio de presion compradora/vendedora, z-score de retornos, sesgo y curtosis.

Las senales de regimen incluyen retornos close-to-close, momentum (Rate of Change), y percentil de volatilidad historica.

---

## 5. Motor de Backtesting

### Arquitectura Event-Driven

El backtester (`engine/backtester.py`, 473 lineas) simula la ejecucion de estrategias barra por barra con modelado realista de costes.

El flujo de ejecucion procede asi: tras un periodo de calentamiento de 50 barras para inicializar indicadores, el backtester itera sobre cada barra. Primero verifica las salidas de posiciones abiertas (stop loss, take profit, time stop, senal contraria). Despues aplica slippage y comision sobre la salida. Luego genera una nueva senal llamando a la funcion de estrategia con el DataFrame, el indice de barra actual y la posicion vigente. A continuacion aplica position sizing mediante el risk manager. Si hay senal y no hay posicion abierta, abre posicion. Finalmente actualiza el equity mark-to-market y registra el estado.

Las posiciones se modelan con precio de entrada ajustado por slippage (adverso: +slip para longs, -slip para shorts), comision deducida inmediatamente del equity, y stops/targets anclados al ATR del momento de entrada.

### Position Sizing

El sistema soporta tres metodos de sizing. El metodo de porcentaje fijo asigna un porcentaje constante del equity por trade. El metodo de escalado por volatilidad (por defecto) calcula el tamano como equity multiplicado por el objetivo de volatilidad dividido por la volatilidad realizada, de modo que posiciones mas grandes en mercados tranquilos y mas pequenas en mercados volatiles. El metodo Kelly utiliza un cuarto del criterio de Kelly basado en la tasa de aciertos estimada.

### Validacion Walk-Forward

El validador walk-forward divide los datos en folds temporales rodantes. Con la configuracion por defecto de 90 dias de entrenamiento, 15 de validacion y 45 de test, se generan 5 folds sobre 365 dias de datos. Los parametros se optimizan en el periodo de entrenamiento, se selecciona la mejor configuracion en validacion, y se mide el rendimiento real en test. La metrica clave es la degradacion OOS: la diferencia entre el Sharpe in-sample y el Sharpe out-of-sample, que debe ser inferior al 50% para considerar la estrategia robusta.

---

## 6. Catalogo de Estrategias

### Estrategias Activas (11)

#### 6.1 Trend Following — Fibonacci Pullback con BMSB

Esta estrategia busca entradas en pullbacks dentro de tendencias establecidas. Utiliza el Bull Market Support Band (SMA-20 por encima de EMA-21) para confirmar la tendencia alcista, y detecta retrocesos a la zona Fibonacci 0.382-0.618 del swing reciente como puntos de entrada optimos. Requiere ADX superior a 30.17 (tendencia fuerte), volumen por encima de 0.98x la media, y RSI entre 34 y 70 para evitar extremos. El stop loss es amplio (4.55x ATR) para dar espacio al trade, con un target de 5.58x ATR. Maximo 48 horas de retencion.

Rendimiento evolutivo: Sharpe 1.62, WF-Sharpe 1.96, 181 trades, $735 de PnL neto.

#### 6.2 Donchian Breakout — Trading Turtle

Inspirada en el sistema Turtle original, esta estrategia entra cuando el precio rompe por encima del maximo de 40 periodos (long) o por debajo del minimo de 40 periodos (short). Requiere confirmacion de volumen superior a 1.5x la media y ADX mayor a 27.6. El target de beneficio es excepcionalmente amplio (7.85x ATR) para capturar breakouts significativos, con un stop de 2.66x ATR. Esta es la estrategia mas robusta del sistema, candidata a produccion.

Rendimiento evolutivo: Sharpe 2.79, WF-Sharpe 2.99, PF 1.89, 132 trades, $1,015 PnL neto. Mejor candidato a produccion.

#### 6.3 Dual MA — Cruce de Medias Moviles

Clasico sistema de cruce entre EMA rapida (20 periodos) y EMA lenta (46 periodos). El golden cross (rapida cruza por encima de lenta) genera senal long, el death cross genera senal short. Filtros adicionales: ADX minimo de 26.55 para confirmar tendencia, volumen superior a 1.07x la media, y opcionalmente precio por encima de la MA lenta para longs. Stops de 3.48x ATR con target de 5.64x ATR. Time stop de 36 horas.

#### 6.4 Keltner Breakout — Canales de Keltner

Entrada cuando el precio rompe por encima o por debajo del canal Keltner (EMA-19 con multiplicador ATR de 3.40). Requiere confirmacion de momentum via RSI (>51 para longs, <45.4 para shorts), ADX minimo de 14.76, y volumen superior a 1.98x la media. Los canales anchos (3.40x ATR) filtran el ruido y solo capturan breakouts genuinos. Stop de 2.61x ATR con target de 6.24x ATR.

#### 6.5 Volatility Squeeze — TTM Squeeze Modificado

Detecta periodos de compresion de volatilidad donde las Bollinger Bands (20, 2.0) se contraen dentro del canal Keltner (20, 1.17x ATR). Cuando esta compresion persiste al menos 3 barras y luego se libera, la estrategia entra en la direccion del momentum (pendiente de regresion lineal de las ultimas 13 barras). Requiere volumen superior a 1.90x la media. Stops de 3.79x ATR con target de 4.60x ATR. Esta es la segunda mejor estrategia, candidata a produccion.

#### 6.6 Ichimoku Kumo — Sistema de Nube Completo

Implementacion completa del sistema Ichimoku con periodos adaptados: Tenkan de 10, Kijun de 34, Senkou B de 59, desplazamiento de 30 barras. La entrada se produce cuando el precio rompe por encima de la nube (Kumo) con confirmacion de cruce Tenkan/Kijun. ADX minimo de 12.12 (filtro suave). Stops de 2.98x ATR con target amplio de 6.82x ATR y time stop generoso de 72 horas para permitir que las tendencias se desarrollen.

#### 6.7 KAMA Trend — Media Movil Adaptativa de Kaufman

La KAMA (Kaufman Adaptive Moving Average) ajusta automaticamente su velocidad segun el Efficiency Ratio del mercado: rapida en tendencias claras, lenta en mercados ruidosos. El ER se calcula sobre 14 barras, con constantes de suavizado rapido de 4 y lento de 26. La entrada ocurre cuando la KAMA gira (cambia de pendiente) o el precio cruza la KAMA con pendiente fuerte (>0.13%). ADX minimo de 16.73. Stops de 2.28x ATR con target de 3.18x ATR.

#### 6.8 Fisher Transform — Oscilador No-Lineal de Ehlers

La transformada Fisher normaliza los precios a una distribucion gaussiana, amplificando los extremos. Se calcula sobre 18 periodos, con suavizado exponencial. La estrategia entra en modo contra-tendencia cuando el Fisher alcanza niveles extremos (>1.98 o <-1.98) y revierte direccion. Requiere ADX menor a 35.32 (evita tendencias extremas donde las reversiones fallan). Stops de 2.18x ATR con target de 3.96x ATR. Time stop corto de 24 horas (es un trade de reversion, no de tendencia).

#### 6.9 Chaos Trend — Exponente de Hurst

Utiliza el exponente de Hurst calculado via analisis R/S (Rescaled Range) multi-escala para identificar regimenes de mercado persistentes. Cuando H esta entre 0.55 y 0.85, el mercado muestra memoria positiva (tendencia), y la estrategia entra en la direccion indicada por el cruce EMA-12/EMA-26 con confirmacion de pendiente de momentum. La dimension fractal D = 2-H debe ser menor a 1.45 (mercado no caotico). Requiere ADX minimo de 20 y volumen 1.2x la media. Stops de 3x ATR con target de 5.5x ATR.

#### 6.10 Vol Regime Arb — Arbitraje de Regimen de Volatilidad

Opera en dos modos basados en la volatilidad Garman-Klass y su z-score rolling. En modo expansion: cuando la volatilidad esta comprimida (z-score < -1.2) durante al menos 5 barras, entra en la direccion del momentum anticipando un breakout. En modo contraccion: cuando la volatilidad es extrema (z-score > 2.0) y el RSI esta en extremo (oversold <30 o overbought >70), fades el movimiento. ADX minimo de 18 para el modo expansion. Stops de 2.5x ATR con target de 4.5x ATR.

#### 6.11 LSTM Pattern — Clustering Multi-Variante + LSTM

La estrategia mas sofisticada del sistema. Combina clustering de patrones de velas con prediccion LSTM de secuencias. El parametro `cluster_variant` es evolucionable entre 7 opciones (kmeans_8/20/50, hier_20/50, bisect_20/50), permitiendo al algoritmo genetico descubrir cual variante funciona mejor.

El pipeline por barra funciona asi: primero se extraen ventanas de 10 velas y se clasifican en clusters. Luego se construyen secuencias de 20 barras con el cluster ID (one-hot) mas 8 features tecnicas. El LSTM predice la probabilidad de direccion (bajada/neutral/subida). La senal final combina la probabilidad del perfil de cluster (peso 0.4) con la prediccion LSTM (peso 0.6), requiriendo una confianza combinada minima del 55%.

Todas las predicciones se pre-computan en batch al inicio del backtest para rendimiento optimo (O(1) por barra en lugar de una llamada a model.predict() por barra).

### Estrategias Podadas (10)

Las siguientes estrategias fueron probadas extensivamente pero no alcanzaron fitness positiva suficiente tras miles de evaluaciones: volatility_breakout (0%), mean_reversion (0%), rsi_divergence (0%), vwap_reversion (0%), obv_divergence (0%), supertrend (0.3%), momentum (0.5%), heikin_ashi_ema (0.4%), connors_rsi2 (1.2%), williams_cci (1.0%). "Podadas" significa que permanecen en el codigo pero estan excluidas del registro de evolucion para no desperdiciar evaluaciones.

---

## 7. Sistema Multi-Clustering

### Arquitectura de 7 Variantes

El modulo `engine/multi_clustering.py` (508 lineas) implementa 7 variantes de clustering que compiten entre si durante la evolucion.

#### Flat K-Means (3 variantes)

K-Means plano con MiniBatchKMeans (batch_size=1024, n_init=10). Tres granularidades: kmeans_8 con 8 clusters (vision gruesa: solo distingue patrones muy diferentes), kmeans_20 con 20 clusters (granularidad media: balance entre especificidad y robustez estadistica), y kmeans_50 con 50 clusters (granularidad fina: captura patrones sutiles pero requiere mas datos para perfiles estables). Esta ultima es la variante baseline.

#### Hierarchical 2-Level (2 variantes)

Clustering jerarquico de dos niveles que imita un arbol genealogico de patrones. El Nivel 1 (macro) clasifica por direccion futura usando percentiles de retorno forward: bearish (percentil <33), neutral (33-67) y bullish (>67), creando 3 macro-grupos. El Nivel 2 (micro) aplica K-Means dentro de cada macro-grupo para subcategorizar por forma del patron. Esto produce clusters como "bullish de impulso rapido" vs "bullish de acumulacion lenta" dentro del mismo grupo alcista.

La variante hier_20 produce 3 macro x 6 micro = 18 clusters efectivos, y hier_50 produce 3 x 16 = 48 clusters efectivos. Para inferencia sin retorno forward (trading real), el macro-grupo se infiere por distancia al centroide mas cercano de cada grupo.

#### BisectingKMeans (2 variantes)

Algoritmo divisivo top-down que comienza con todos los datos en un solo cluster y lo divide iterativamente. En cada paso, el cluster mas grande se biseca en dos usando K-Means con k=2. Esto genera un arbol binario natural donde las relaciones padre-hijo se preservan internamente, y cada cluster refina progresivamente a su antecesor. bisect_20 produce 20 clusters hoja y bisect_50 produce 50.

### Ingenieria de Features de Patrones

El modulo `engine/pattern_clustering.py` (572 lineas) transforma ventanas deslizantes de 10 velas en vectores de 100 dimensiones. Cada vela se normaliza de forma scale-invariant usando el primer cierre y el rango del patron, y se extraen 10 features por vela: open/high/low/close normalizados, volumen normalizado, cuerpo de la vela, mecha superior, mecha inferior, rango total y retorno.

Los perfiles de cluster se construyen midiendo los retornos forward (siguientes 5 barras) de todas las ventanas asignadas a cada cluster, derivando probabilidad de subida, probabilidad de bajada, media de retorno, Sharpe, win rate y ratio ganancia/perdida promedio. Estos perfiles se actualizan incrementalmente con decaimiento exponencial (factor 0.85) para adaptarse a cambios de regimen.

---

## 8. Modelo LSTM de Patrones

### Arquitectura de Red

El modelo LSTM (`engine/lstm_pattern_model.py`, 416 lineas) procesa secuencias de patrones para predecir la direccion del mercado.

La entrada es un tensor de forma (batch, 20, N_features) donde N_features = N_clusters + 8 features tecnicas. Por ejemplo, para kmeans_50: 50 (one-hot cluster ID) + 8 (RSI, ADX, volume_ratio, return_zscore, BB_%B, pressure_imbalance, GK_vol, momentum) = 58 dimensiones.

La arquitectura consiste en una capa LSTM de 48 unidades con return_sequences=True, seguida de Dropout al 30%, otra capa LSTM de 24 unidades, otro Dropout al 30%, una capa densa de 32 neuronas con activacion ReLU, y dos cabezas de salida: una cabeza de direccion con 3 salidas softmax (P(bajada), P(neutral), P(subida)) y una cabeza de magnitud con 1 salida lineal (retorno esperado absoluto).

### Entrenamiento

El modelo se entrena con sparse_categorical_crossentropy para direccion y MSE para magnitud, con pesos de 1.0 y 0.3 respectivamente. El balanceo de clases se implementa via sample_weight por muestra (no class_weight, que Keras no soporta en modelos multi-output). La validacion es por split temporal (ultimo 15% como OOS). Los modelos alcanzan aproximadamente 60% de precision direccional en validacion, vs un baseline del 33%.

### Prediccion en Batch

Para rendimiento optimo durante backtesting, todas las predicciones LSTM se pre-computan en una sola llamada a model.predict() al inicio. Esto transforma el coste de O(n * 10-50ms) (una prediccion por barra) a O(1) con un unico batch de 2-3 segundos para ~8,760 barras. Las predicciones se almacenan en un array indexado por barra, y la funcion de estrategia simplemente consulta el array.

---

## 9. Motor de Evolucion Genetica

### Representacion del Genoma

Cada individuo de la poblacion es un Genome que contiene el nombre de la estrategia, un diccionario de parametros, el ID de generacion, las IDs de los padres (para trazabilidad genealogica), y las metricas de fitness (Sharpe, profit factor, trades, drawdown).

### Registro de Estrategias

El STRATEGY_REGISTRY define las 11 estrategias activas con su modulo, funcion, diccionario de parametros y espacio de busqueda. Cada parametro tiene un tipo (float, int, choice, bool), un rango minimo-maximo, y opcionalmente un paso. El parametro `cluster_variant` de lstm_pattern es de tipo "choice" con 7 opciones, permitiendo al GA seleccionar la variante de clustering optima.

### Ciclo Evolutivo

La evolucion procede en generaciones de 40 individuos. La generacion 0 se inicializa con 85% aleatorio (exploracion amplia) y 15% round-robin entre estrategias (representacion garantizada). En cada generacion subsiguiente, la seleccion por torneo (k=3) elige padres, el crossover uniforme (p=0.5) mezcla parametros cuando ambos padres comparten estrategia, la mutacion gaussiana (p=0.3, sigma=15% del rango) introduce variacion, y un 15% de la poblacion se reemplaza con individuos frescos aleatorios para prevenir convergencia prematura.

### Funcion de Fitness

La fitness se calcula como el Sharpe ratio del walk-forward, penalizado por drawdown excesivo (-0.5 por cada punto porcentual de DD sobre 15%), penalizado por degradacion OOS (-0.3 por cada punto sobre 30%), y escalado por profit factor (debe superar 1.3 para puntuacion completa).

### Hall of Fame

Se mantienen los 4 mejores genomas por estrategia con diversidad forzada: se rechaza un nuevo candidato si su distancia euclidiana normalizada respecto a los existentes es menor a 0.10. Esto garantiza que el Hall of Fame contiene soluciones diversas, no variaciones triviales del mismo optimo local.

### Paralelizacion

La evaluacion de genomas se distribuye entre multiples procesos via ProcessPoolExecutor con contexto spawn (necesario en macOS). Cada worker inicializa un entorno aislado: deshabilita CUDA, deshabilita Metal GPU en Apple Silicon (via tf.config.set_visible_devices([], 'GPU')), y carga los datos una sola vez desde un pickle compartido. Esto permite evaluar 8-12 genomas en paralelo en un M2 Pro.

### Ejecucion Tipica (14 horas)

En las generaciones 0-5 (primera hora) se realizan ~3,000 evaluaciones con mejoras rapidas de 0.1-0.2 puntos de Sharpe por generacion. En las generaciones 6-30 (horas 1-8) la mejora se desacelera a 0.05/generacion, con elites alcanzando Sharpe 1.5-2.0 y degradacion OOS del 20-30%. En las generaciones 31-50+ (horas 8-14) los retornos son marginales (~0.01/generacion) y la exploracion se intensifica. El resultado final son 40-44 genomas elite con donchian_breakout liderando a Sharpe 2.79 y volatility_squeeze en segundo lugar a 2.15.

---

## 10. Metricas y Evaluacion

### Metricas de Rendimiento

El modulo `engine/metrics.py` (267 lineas) computa un conjunto exhaustivo de metricas. Las metricas de rendimiento incluyen retorno total y anualizado, ratios Sharpe, Sortino y Calmar, drawdown maximo y su duracion, volatilidad anualizada, desviacion a la baja, Value-at-Risk al 95% y Conditional VaR al 95%.

Las estadisticas de trades cubren numero total, tasa de aciertos, profit factor, expectativa (PnL promedio por trade), promedio de ganancias y perdidas, perdidas consecutivas maximas, y horas promedio de retencion.

El analisis de costes desglosa comisiones totales (0.1% en entrada y salida), slippage total (5 bps promedio), y la diferencia entre PnL bruto y neto.

Las metricas de robustez incluyen Sharpe OOS (walk-forward out-of-sample), degradacion OOS porcentual, Sharpe medio y desviacion en cross-asset testing (BTC, ETH, SOL, ADA), y estabilidad por regimen.

---

## 11. Gestion de Riesgo

### RiskManager

El modulo `engine/risk_manager.py` (174 lineas) implementa tres capas de proteccion.

La primera capa es el position sizing, que escala el tamano de cada posicion para apuntar a una volatilidad anualizada del 30%. Esto significa posiciones grandes en mercados tranquilos y pequenas en mercados volatiles, manteniendo un perfil de riesgo constante.

La segunda capa son los circuit breakers: un freno automatico si la perdida diaria alcanza el 3% del equity, y un freno total si el drawdown acumulado alcanza el 10%. Estos mecanismos detienen completamente la operativa hasta que se reseteen manualmente, previniendo perdidas catarstroficas.

La tercera capa son los limites de posicion: maximo 3 posiciones concurrentes, intervalo minimo de 15 minutos entre trades (anti-overtrading), y retencion maxima de 48 horas por defecto (configurable por estrategia).

### Stops y Targets

Cada trade tiene un stop loss basado en ATR (tipicamente 2-4.5x), un take profit basado en ATR (tipicamente 3-8x), y un time stop configurable. La estrategia puede sobreescribir los defaults por senal, y el backtester los aplica en cada barra verificando si el high/low del bar toca los niveles.

---

## 12. Simulacion Monte Carlo

### Tres Metodos de Bootstrap

El modulo `engine/monte_carlo.py` (303 lineas) valida la robustez estadistica de los resultados.

El metodo de permutacion de trades reordena aleatoriamente la secuencia de trades manteniendo los mismos PnL individuales. Si los resultados dependen del orden (por ejemplo, los primeros trades fueron afortunados), esto lo detecta. Se ejecutan 1,000 permutaciones y se computa el percentil 5% del Sharpe como limite inferior de confianza.

El bootstrap de retornos remuestrea retornos diarios con reemplazo para generar curvas de equity sinteticas. Esto estima el intervalo de confianza del Sharpe sin asumir distribucion normal.

El block bootstrap preserva la correlacion serial (clustering de volatilidad) remuestreando bloques contiguos de retornos. Es mas realista que el bootstrap simple para series temporales financieras donde los dias volatiles tienden a agruparse.

---

## 13. Deteccion de Regimen

### Clasificacion de Mercado

El sistema clasifica cada periodo de mercado en uno de seis regimenes: TREND (ADX alto con direccionalidad), BREAKOUT (expansion de volatilidad), MEAN_REVERSION (ADX bajo con extremos de RSI), LATERAL (ADX bajo sin senal clara), EXOGENOUS_SHOCK (movimiento anomalo), y UNKNOWN.

La deteccion se basa en el ADX para fuerza de tendencia, el percentil de bandwidth de Bollinger para compresion/expansion de volatilidad, y el DI+/DI- para direccionalidad. Las estrategias respetan su regimen asignado: Trend Following solo opera en TREND, Donchian en BREAKOUT+TREND, Fisher Transform en RANGE, y Vol Squeeze en la transicion de squeeze a breakout.

---

## 14. Pipeline de Investigacion Autonoma

### AutoResearch

El sistema incluye un loop de investigacion autonoma inspirado en el modelo de Andrej Karpathy. El agente genera hipotesis de trading, las formaliza como mutaciones de un solo parametro, ejecuta backtests walk-forward, compara contra el baseline, y acepta o rechaza con documentacion del razonamiento. La regla es estricta: una sola variable por experimento, validacion walk-forward obligatoria, y aceptacion solo si la mejora es mayor o igual al 2% con metricas de guardia estables.

### MiroFish

El motor de escenarios MiroFish simula 7 agentes que analizan el mercado desde diferentes angulos: tecnico, fundamental, sentiment, on-chain, macro, flujo de ordenes y correlacion cross-asset. El consenso de los agentes genera hipotesis priorizadas que alimentan al AutoResearch.

---

## 15. PineScript — Equivalente para TradingView

### Implementacion v2

El sistema incluye un equivalente PineScript v6 de 508 lineas que replica 9 de las 11 estrategias (excluye LSTM Pattern por requerir modelo ML, y Chaos Trend por aproximacion insuficiente del exponente de Hurst en PineScript).

Las mejoras clave de v2 sobre v1 incluyen: deteccion de regimen (ADX + bandwidth + DI para clasificar TREND/BREAKOUT/RANGE), senales ponderadas (cada estrategia contribuye un peso configurable en lugar de un voto binario), confirmacion de timeframe superior (EMA50 vs EMA200 en 4H), trailing stop (se activa despues de 1.5x ATR de profit, persigue a 1.8x ATR), cooldown post-perdida (3 barras de espera), y un dashboard visual con estado de cada estrategia, regimen actual, scores y drawdown.

---

## 16. Flujo de Datos End-to-End

El pipeline completo comienza en la API de Binance via CCXT, pasa por el cache Parquet, entra en la ingenieria de features (35 indicadores), alimenta el backtester event-driven donde cada estrategia genera senales, el risk manager dimensiona posiciones, y se ejecutan trades simulados. Los resultados pasan por el modulo de metricas para computar Sharpe, Sortino, drawdown, profit factor y demas. El validador walk-forward repite este proceso en 5 folds temporales. Los resultados alimentan la funcion de fitness del algoritmo genetico, que selecciona, cruza, muta y genera nuevas generaciones. Los mejores genomas se almacenan en el Hall of Fame con diversidad forzada. Opcionalmente, los resultados pasan por simulacion Monte Carlo para validacion estadistica.

Para el pipeline de clustering+LSTM, los datos OHLCV se transforman en ventanas de 10 velas, se normalizan, se clusterizan (7 variantes), se construyen perfiles de cluster, se entrenan los LSTM por variante, y los modelos resultantes se guardan para que la estrategia lstm_pattern los utilice durante la evolucion.

---

## 17. Infraestructura Tecnica

### Requisitos de Hardware

El sistema esta optimizado para Apple Silicon M2 Pro con GPU Metal. La paralelizacion usa ProcessPoolExecutor con contexto spawn (obligatorio en macOS). Los workers deshabilitan la GPU Metal para evitar contension, operando solo en CPU. El entorno virtual Python gestiona las dependencias, con pip instalando contra system packages.

### Almacenamiento

Los datos OHLCV en Parquet ocupan tipicamente 50-200 MB dependiendo de la resolucion temporal. Los modelos (70+ archivos entre clustering y LSTM) ocupan aproximadamente 500 MB. Los logs y reportes de evolucion crecen linealmente con las horas de ejecucion.

### Ejecucion

El comando principal para evolucion es `nohup bash train.sh --evolve --evolve-hours 10 > evolution.log 2>&1 &`, que lanza el proceso en background con output redirigido a log. El script `train.sh` gestiona todos los modos: full pipeline, solo modelo TF, solo pipeline de investigacion, walk-forward, optimizacion paralela y evolucion genetica.

---

## 18. Resultados Clave

### Mejores Estrategias Evolucionadas

Donchian Breakout lidera con Sharpe 2.79, WF-Sharpe 2.99, profit factor 1.89, 132 trades y $1,015 de PnL neto. Es la estrategia mas consistente y robusta del sistema.

Volatility Squeeze ocupa el segundo lugar con Sharpe 2.15, WF-Sharpe 1.95, y buena consistencia across folds.

Trend Following es la tercera mejor con Sharpe 1.62, WF-Sharpe 1.96, mostrando buena robustez OOS a pesar de un Sharpe in-sample modesto.

### Tasa de Exito Evolutivo

De 21 estrategias probadas, 11 mostraron fitness positiva y se mantienen activas. 10 fueron podadas tras miles de evaluaciones sin alcanzar viabilidad. Los resultados muestran que las estrategias de breakout y squeeze son significativamente superiores a las de mean-reversion en XRP/USDT, probablemente debido a la naturaleza trending del activo en los periodos analizados.

---

## 19. Proximos Pasos

Las areas de desarrollo futuro incluyen: despliegue en paper trading con ordenes reales en Binance Testnet, extension a multiples pares (BTC, ETH, SOL), ensemble de estrategias con ponderacion por riesgo (risk parity), integracion de datos on-chain y sentiment para enriquecer features, y optimizacion continua del pipeline LSTM con nuevas arquitecturas (Transformers, attention mechanisms).
