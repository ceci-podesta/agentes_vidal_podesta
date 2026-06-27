"""Archivo de configuración de pytest a nivel raíz del proyecto.

Su sola presencia hace que pytest agregue la raíz del proyecto al
`sys.path`, de modo que `import mia_agents` e `import student_framework`
funcionen al ejecutar `pytest` (sin necesidad de `python -m pytest`).
"""
