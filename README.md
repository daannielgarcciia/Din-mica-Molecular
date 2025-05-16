# Dinámica-Molecular
Simulación de Dinámica Molecular: Argón Líquido

Este repositorio contiene el código fuente en Python (dinamica_molecular.py) para una simulación de dinámica molecular (DM) de argón líquido aislado, desarrollada como parte de un proyecto académico para el curso de Física Experimental y Computacional II. El objetivo es estudiar las propiedades termodinámicas del argón líquido, como temperatura, presión, y energías cinética y potencial, utilizando el potencial de Lennard-Jones y el algoritmo de Verlet.

Descripción
El código simula un sistema de partículas de argón líquido en una caja cúbica con condiciones periódicas de contorno. Se implementan tres casos:

Simulación principal: ( N = 1004 ) partículas, paso temporal ( \Delta t = 10^{-14} ) s, 1500 pasos.
Caso reducido: ( N = 27 ) partículas, mismo ( \Delta t ), 500 pasos.
Paso temporal grande: ( N = 1004 ), ( \Delta t = 10^{-9} ) s, 500 pasos.

El sistema se inicializa con partículas equiespaciadas en las aristas de la caja, velocidades aleatorias escaladas a ( T_0 = 120 , \text{K} ), y densidad ( \rho = 1680 , \text{kg/m}^3 ). Se calculan propiedades termodinámicas y se generan gráficos y animaciones para visualizar la evolución del sistema.
Requisitos

Python 3.8+
Bibliotecas necesarias:pip install numpy matplotlib pandas numba


Instalación

Clona el repositorio:
git clone https://github.com/daannielgarcciia/Din-mica-Molecular.git
cd dinamica_molecular


Instala las dependencias:
pip install -r requirements.txt


(Opcional) Crea un archivo requirements.txt con:
numpy
matplotlib
pandas
numba



Uso
Ejecuta el script principal:
python dinamica_molecular.py

El código generará:

Archivos Excel con resultados numéricos:
resultados_N1004.xlsx (( N = 1004 ))
Medias_y_varianza_N1004.xlsx (( N = 1004 ))
resultados_N27.xlsx (( N = 27 ))
Medias_y_varianza_N27.xlsx (( N = 27 ))
resultados_dt_large.xlsx (( \Delta t = 10^{-9} ) s)
Medias_y_varianza_dt_large.xlsx (( \Delta t = 10^{-9} ) s)


Gráficos (PNG):
Distribuciones de partículas en instantes clave.
Evolución temporal de energías, temperatura, presión, y velocidad del centro de masa.
Trayectorias y velocidades de 6 partículas seleccionadas.
Errores estadísticos vs tamaño de bloque.


Animaciones (GIF):
animacion_3d_N1004.gif, animacion_3d_N27.gif, animacion_3d_dt_large.gif



Los archivos generados estarán en el directorio de ejecución. Enlaces a los resultados:
https://drive.google.com/drive/folders/10sEyEKgoUNTWIKq-ZpeuoDQxAihP1oCB?usp=drive_link

Resultados Excel
Animaciones

Estructura del Código

Parámetros físicos: Definición de constantes (( N ), ( \sigma ), ( T_0 ), ( \rho ), etc.) y magnitudes adimensionales.
Funciones principales:
init_posiciones: Coloca partículas en las aristas de la caja.
init_velocidades: Genera velocidades aleatorias ajustadas.
calcular_fuerzas: Computa fuerzas, potencial y virial (optimizado con numba).
verlet: Integra el movimiento con el algoritmo de Verlet.
generar_graficas: Produce visualizaciones y animaciones.


Simulaciones: Ejecuta las tres configuraciones, guarda datos y genera gráficos.

Resultados

Simulación principal (( N = 1004 )): Resultados estables, con fluctuaciones pequeñas en energía total (( 2.3 \times 10^{-3} )), temperatura (120 K), y presión (10^7 Pa).
Caso ( N = 27 ): Mayores fluctuaciones (( 1.8 \times 10^{-2} )) debido al menor número de partículas, pero cualitativamente similar.
Caso ( \Delta t = 10^{-9} ) s: Inestabilidad numérica, con energías y temperaturas divergentes, indicando un paso temporal inadecuado.

Los gráficos muestran:

Distribuciones de partículas evolucionando hacia el equilibrio.
Conservación de energía (excepto en ( \Delta t = 10^{-9} ) s).
Trayectorias aleatorias y velocidades fluctuantes, típicas de un líquido.
Errores estabilizados para bloques de tamaño ( \geq 5 ).
