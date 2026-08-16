# Diseño: tokens de color y tema declarado en el dashboard

**Fecha:** 2026-08-16
**Estado:** pendiente de revisión
**Entrega:** 4 de 4 del bloque de front (UX del Run → Quality Gate → tendencias y
exportación → **coherencia visual**)

## 1. Resumen

El dashboard pinta su color en **328 literales hexadecimales** repartidos por 11
ficheros. Medido sobre main: **301 de ellos son repeticiones de unos 15 colores**
(`#e2e8f0` aparece 41 veces, `#94a3b8` y `#6366f1` 30 veces cada uno...) y solo 27 se
usan una única vez. Cambiar un color del sistema hoy significa una búsqueda global y
confiar en no olvidar ninguno; el futuro tema claro sería inabordable sobre esta base.

Además no existe `.streamlit/config.toml`, así que el chrome de Streamlit (barra
superior, menús) usa su tema por defecto mientras el CSS del proyecto pinta el cuerpo en
oscuro. El desajuste es visible en cualquier captura: franja clara con el botón *Deploy*
sobre un cuerpo oscuro.

Esta entrega extrae los colores repetidos a una única fuente de verdad y declara el tema
a Streamlit. **No cambia el aspecto del cuerpo del dashboard en un solo píxel** — esa es
precisamente su definición de correcto — salvo el chrome, que cambia a propósito.

## 2. Objetivos y no-objetivos

### Objetivos

- `src/dashboard/components/theme.py` como única fuente de verdad del color: paleta
  semántica más las escalas por categoría y severidad que hoy viven en `charts.py`.
- Todo color usado **más de una vez** pasa a token. Fase verificada por comparación de
  píxeles: cualquier diferencia en el cuerpo de una página es un fallo.
- `.streamlit/config.toml` declarando el tema oscuro con los valores de la paleta, para
  que el chrome deje de desentonar.
- Un test guardián que impide la reaparición de literales repetidos fuera de `theme.py`.

### No-objetivos

- **Tema claro.** Queda explícitamente fuera; esta entrega lo deja a un cambio de
  valores, no de estructura. Los gráficos de Plotly necesitarían además su propia paleta
  alternativa (Plotly no lee CSS), lo que confirma que es una entrega aparte.
- Rediseño visual de ningún tipo. Ni un espaciado, ni una fuente, ni un radio de borde.
- Tokenizar los 27 colores de un solo uso. Bautizarlos sería inventar nombres para
  colores que no forman parte del sistema; el test guardián los vigila si algún día se
  repiten.
- Modo de alto contraste o accesibilidad de color (auditoría aparte si se quiere).

## 3. Diseño

### `theme.py`

```python
PALETTE: dict[str, str]          # ~15 colores semánticos: text, text_muted, accent, ...
CATEGORY_COLORS: dict[str, str]  # movido desde charts.py
SEVERITY_COLORS: dict[str, str]  # movido desde charts.py
```

Consumo en dos frentes, porque hay dos consumidores con capacidades distintas:

- **Código Python** (f-strings de HTML, Plotly, helpers como `pass_rate_color`): importa
  `PALETTE` y usa sus valores directamente. Plotly no puede leer variables CSS, así que
  esta vía es obligatoria para los gráficos en cualquier caso.
- **El bloque CSS estático** (`DESIGN_CSS`, ~370 líneas en `styles.py`): se convierte en
  plantilla sobre la misma `PALETTE`.

**Decisión que cambia respecto a la propuesta en conversación:** nada de variables CSS
(`:root { --token }`). Habrían sido un segundo mecanismo que Plotly no puede consumir,
así que no habilitan nada y duplican la indirección. La plantilla se resuelve con
`string.Template` (`$token`) y no con f-strings: el CSS está lleno de llaves y un
f-string obligaría a duplicarlas todas — un cambio masivo y propenso a errores en un
refactor cuyo listón es "ni un píxel".

### Compatibilidad

`charts.py` re-exporta `COLORS`, `CATEGORY_COLORS` y `SEVERITY_COLORS` como alias de
`theme` para no romper a las páginas que hoy los importan de ahí. De paso se corrige un
acoplamiento raro: `2_results.py` y `3_compare.py` importan colores desde el módulo de
*gráficos* para pintar HTML; pasan a importar de `theme`.

### Criterio de sustitución

Un literal se sustituye si su color (normalizado a minúsculas) aparece **más de una vez
en todo `src/dashboard/`**. Uno por fichero pero dos en total cuenta como repetido: el
guardián mide el sistema, no el fichero.

### `.streamlit/config.toml`

`[theme]` con `base = "dark"` y `primaryColor` / `backgroundColor` /
`secondaryBackgroundColor` / `textColor` tomados de `PALETTE`. Es la segunda fase, tras
el refactor de tokens, porque su verificación es opuesta: aquí el chrome **debe**
cambiar, y mezclarlo con la fase de píxel idéntico invalidaría la comparación.

## 4. Verificación

### Fase 1 — tokens: comparación del estilo aplicado

1. Con main, capturar el cuerpo (`stMain`) de las 6 vistas (Home, Run, Results, Compare,
   Test Cases, Gate) a viewport fijo, esperando al contenido para evitar estados de
   carga. De cada una se guardan dos cosas: el PNG y el HTML renderizado.
2. Aplicar el refactor y repetir la captura en las mismas condiciones.
3. Comparar.

**Corregido tras medirlo.** El plan original era comparar píxeles con el criterio
«cualquier diferencia es un fallo». Ese criterio no se sostiene: capturando dos veces
*sin tocar el código*, hasta 74 píxeles por página difieren por antialiasing de bordes
redondeados y reposicionamiento subpíxel de las anotaciones de Plotly. Un umbral de
tolerancia sería subjetivo justo donde hace falta precisión.

La medida exacta es otra: **de cada página se extraen los atributos `style`, los
atributos de pintado SVG (`fill`, `stroke`, `stop-color`) y las reglas CSS inyectadas,
se ordenan y se comparan como texto**. Eso es exactamente lo que este refactor puede
romper, y no tiene ruido. Comparar el DOM entero tampoco vale: Plotly genera hashes por
figura, BaseWeb lleva un contador global y las clases de emotion cambian entre sesiones.

Criterio: **el estilo aplicado debe ser idéntico carácter a carácter en las 6 vistas**.
Los píxeles se siguen comparando como señal secundaria, y cualquier diferencia se
inspecciona visualmente antes de descartarla.

Validación del propio instrumento: antes de refactorizar nada, se captura dos veces el
mismo código y debe dar idéntico. Sin ese control, un arnés silencioso pasaría por
«sin cambios».

El arnés es utillaje de scratchpad, no se commitea: necesita servidor en marcha y
navegador, y no es ejecutable en CI. La evidencia se documenta en la PR.

### Fase 2 — config.toml: a ojo

El chrome (barra superior, menús, fondo del sidebar nativo) debe verse integrado con el
cuerpo. Captura antes/después en la PR.

### Test guardián

En `tests/test_dashboard_theme.py`: escanea `src/dashboard/**/*.py` (excluyendo
`theme.py`), recoge los literales `#rrggbb` normalizados y falla en dos supuestos:

- **el literal aparece más de una vez** en el conjunto — repetición es sistema, y el
  sistema vive en `theme.py`;
- **el literal coincide con un valor de `PALETTE`** aunque aparezca una sola vez — si el
  color ya tiene nombre, escribir su hex a mano es el despiste exacto que este test
  existe para cazar.

Protege el refactor de su modo de degradación natural: sin él, en tres meses vuelve a
haber 328. Los colores de un solo uso ajenos a la paleta pasan; si alguien reutiliza
uno, el test le obliga a tokenizarlo.

También: tests de que `DESIGN_CSS` resuelto no contiene `$` sin sustituir (una plantilla
a medio resolver se serviría como CSS roto y silencioso) y de que los alias de
`charts.py` siguen apuntando a `theme`.

## 5. Estrategia de tests (TDD)

El guardián y los tests de plantilla se escriben primero y **deben fallar sobre main**
(el guardián, con los 301 repetidos actuales, falla estrepitosamente — esa es su prueba
de RED). La suite completa existente (439 tests) debe seguir verde sin tocar ninguno: si
un test existente se rompe, el refactor cambió comportamiento y no debía.

## 6. Riesgos

- **El refactor toca 11 ficheros sin test visual en CI.** Mitigado por la comparación de
  píxeles en el momento del cambio y por el guardián después. La ventana sin protección
  es el futuro entre PRs, y ahí el guardián es lo único automático — por eso se commitea
  él y no el arnés.
- **`string.Template` a medio sustituir.** Cubierto por el test de `$` residual.
- **`config.toml` cambia también cosas del cuerpo** (Streamlit deriva algunos colores de
  widgets del tema). Se acepta si el resultado es más coherente; si algún widget queda
  ilegible, se ajusta el CSS del proyecto en esa misma fase, nunca en la fase 1.
