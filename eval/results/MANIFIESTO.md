# Reportes históricos M3

Estos JSON se reconstruyeron desde `prints primeras corridas.md` y corresponden
a los experimentos que están documentados en `evolución_agente_M3.md`.

## Incluidos

- Baseline de `color-locks` con prompt genérico y `max_iterations=10`.
- Corridas 1 a 8 de `evolución_agente_M3.md`.

## Excluido

La sección **Corrida 6** de `prints primeras corridas.md` no se exportó. Es una
prueba exploratoria de cuatro escenarios que no tiene configuración ni decisión
asociada en `evolución_agente_M3.md`; incluirla como evidencia final confundiría
la secuencia experimental.

Las corridas 9 a 11 ya se generan de forma nativa con el runner actual en
`eval/results/development/` y no deben reconstruirse desde prints.
