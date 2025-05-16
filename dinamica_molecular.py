import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from numba import njit

### ============================= PARÁMETROS FÍSICOS ============================= ###
N = 12 * 83 + 8  # ~10^3 partículas
sigma = 3.4e-10
T0 = 120
A = 0.039948
NA = 6.022e23
m = A / NA
kB = 1.3806e-23
epsilon = kB * T0
rho = 1680
dt = 1e-14

### ============================= MAGNITUDES ADIMENSIONALES ============================= ###
L = ((N * m) / rho) ** (1 / 3)
L_adi = L / sigma
dt_adi = dt * np.sqrt(epsilon / (m * sigma ** 2))
rc = L_adi / 2
rc_2 = rc ** 2
V = (L_adi * sigma) ** 3

intervalo = 1  # Intervalo para guardar datos

### ============================= FUNCIONES ============================= ###

# Inicialización de partículas en las aristas del cubo
def init_posiciones(N, L):
    part_vertice = N // 12
    resto = N % 12
    pos = []
    edges = [
        ([0, 0, 0], [L, 0, 0]), ([0, 0, L], [L, 0, L]), ([0, L, 0], [L, L, 0]), ([0, L, L], [L, L, L]),
        ([0, 0, 0], [0, L, 0]), ([L, 0, 0], [L, L, 0]), ([0, 0, L], [0, L, L]), ([L, 0, L], [L, L, L]),
        ([0, 0, 0], [0, 0, L]), ([L, 0, 0], [L, 0, L]), ([0, L, 0], [0, L, L]), ([L, L, 0], [L, L, L]),
    ]
    for edge_idx, (inicio, fin) in enumerate(edges):
        inicio = np.array(inicio) - L / 2
        fin = np.array(fin) - L / 2
        n_part = part_vertice + 1 if edge_idx < resto else part_vertice
        for i in range(n_part):
            t = (i + 0.5) / n_part
            pos_xyz = inicio + t * (fin - inicio)
            pos.append(pos_xyz.tolist())
    return np.ascontiguousarray(pos[:N], dtype=np.float64)

# Inicialización de velocidades
def init_velocidades(N, T0, m):
    v = np.random.rand(N, 3) - 0.5
    v -= np.mean(v, axis=0)
    v_rms = np.sqrt(3 * kB * T0 / m)
    v *= v_rms / np.sqrt(np.mean(np.sum(v**2, axis=1)))
    return v

# Cálculo de la partícula imagen más cercana
@njit
def min_imagen(rij, L):
    return rij - L * np.rint(rij / L)

# Fuerza de Lennard-Jones
@njit
def fuerza_LJ(r):
    if r == 0 or r > rc:
        return 0.0
    r6 = (1 / r)**6
    r12 = r6**2
    return 48 * (r12 - 0.5 * r6) / r

# Potencial de Lennard-Jones
@njit
def potencial_LJ(r):
    if r == 0 or r > rc:
        return 0.0
    r6 = (1 / r)**6
    r12 = r6**2
    return 4 * (r12 - r6)

# Cálculo de fuerzas
@njit
def calcular_fuerzas(pos, L):
    N = pos.shape[0]
    F = np.zeros((N, 3))
    U = 0.0
    virial = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = min_imagen(pos[j] - pos[i], L)
            r_sq = np.dot(rij, rij)
            if r_sq < 0.9:
                r_sq = 0.9
            if r_sq < rc_2:
                r = np.sqrt(r_sq)
                fij = fuerza_LJ(r) * rij / r
                F[i] += fij
                F[j] -= fij
                U += potencial_LJ(r)
                virial += (1 / r**6)**2 - 0.5 * (1 / r**6) * r
    return F, U, virial

# Algoritmo de Verlet
@njit
def verlet(pos, vel, F, dt, L):
    a = F
    pos_new = pos + vel * dt + 0.5 * a * dt**2
    pos_new = (pos_new + L / 2) % L - L / 2
    F_new, U, virial = calcular_fuerzas(pos_new, L)
    a_new = F_new
    vel_new = vel + 0.5 * (a + a_new) * dt
    return pos_new, vel_new, F_new, U, virial

# Energía cinética
def energia_cinetica(vel, m, N):
    return 0.5 * m * np.sum(vel**2) / N

# Temperatura
def temperatura(vel, m, N):
    return np.sum(m * vel**2) / (3 * N * kB)

# Presión
def presion(T, virial, V, N):
    return (N * kB * T + (16 / 3) * virial * epsilon) / V

# Velocidad del centro de masa
def velocidad_cm(vel):
    return np.mean(vel, axis=0)

# Análisis de bloques
def block_analisis(data, block_sizes):
    varianzas = []
    for block_size in block_sizes:
        n = len(data)
        n_blocks = n // block_size
        if n_blocks <= 1:
            varianzas.append(0.0)
            continue
        blocks = [data.iloc[i:i + block_size] for i in range(0, len(data), block_size)]
        block_means = np.array([np.mean(block) for block in blocks])
        varianza = np.sum((block_means - np.mean(block_means))**2) / (n_blocks - 1)
        varianzas.append(varianza)
    return varianzas

# Graficar distribución de partículas
def plot_distribucion(pos, L, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_zlim(-L/2, L/2)
    ax.set_xlabel('X*')
    ax.set_ylabel('Y*')
    ax.set_zlabel('Z*')
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

# Graficar trayectorias de partículas seleccionadas
def plot_trayectorias(particulas_pos, L, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(particulas_pos)):
        pos = particulas_pos[i]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f'Partícula {i+1}')
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_zlim(-L/2, L/2)
    ax.set_xlabel('X*')
    ax.set_ylabel('Y*')
    ax.set_zlabel('Z*')
    ax.set_title('Trayectorias de 6 Partículas')
    ax.legend()
    plt.savefig(filename)
    plt.close()

# Graficar componentes de velocidad
def plot_velocidades(particulas_vel, tiempo, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(len(particulas_vel)):
        vel = particulas_vel[i]
        ax1.plot(tiempo, vel[:, 0], label=f'Partícula {i+1}')
        ax2.plot(tiempo, vel[:, 1], label=f'Partícula {i+1}')
        ax3.plot(tiempo, vel[:, 2], label=f'Partícula {i+1}')
    ax1.set_ylabel('v_x (m/s)')
    ax2.set_ylabel('v_y (m/s)')
    ax3.set_ylabel('v_z (m/s)')
    ax3.set_xlabel('Tiempo (ps)')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.suptitle('Componentes de Velocidad de 6 Partículas')
    plt.savefig(filename)
    plt.close()

# Función para generar todas las gráficas
def generar_graficas(Ek_list, Ep_list, Et_list, T_list, P_list, vcm_list, particulas_pos, particulas_vel, instantes, tiempo, L_adi, steps, block_sizes, name):
    # Energías vs Tiempo
    plt.figure()
    plt.plot(tiempo, Ek_list, label='E. Cinética')
    plt.plot(tiempo, Ep_list, label='E. Potencial')
    plt.plot(tiempo, Et_list, label='E. Total')
    plt.xlabel('Tiempo (ps)')
    plt.ylabel('Energía (J/partícula)')
    plt.title('Energías vs Tiempo')
    plt.legend()
    plt.grid()
    plt.savefig(f"energias_vs_tiempo_{name}.png")
    plt.close()

    # Temperatura y Presión vs Tiempo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(tiempo, T_list, label='Temperatura')
    ax1.set_ylabel("Temperatura (K)")
    ax1.grid()
    ax2.plot(tiempo, P_list, label='Presión', color='orange')
    ax2.set_xlabel("Tiempo (ps)")
    ax2.set_ylabel("Presión (Pa)")
    ax2.grid()
    plt.savefig(f"temperatura_presion_vs_tiempo_{name}.png")
    plt.close()

    # Velocidad del Centro de Masas
    plt.figure()
    plt.plot(tiempo, vcm_list)
    plt.xlabel("Tiempo (ps)")
    plt.ylabel("|v_CM| (m/s)")
    plt.title("Velocidad del Centro de Masas")
    plt.grid()
    plt.savefig(f"velocidad_cm_{name}.png")
    plt.close()

    # Error vs Tamaño de Bloque
    df_temp = pd.DataFrame({"E_k": Ek_list, "E_p": Ep_list, "E_t": Et_list, "T": T_list, "P": P_list})
    Ek_varianzas = block_analisis(df_temp['E_k'], block_sizes)
    Ep_varianzas = block_analisis(df_temp['E_p'], block_sizes)
    Et_varianzas = block_analisis(df_temp['E_t'], block_sizes)
    T_varianzas = block_analisis(df_temp['T'], block_sizes)
    P_varianzas = block_analisis(df_temp['P'], block_sizes)
    plt.figure(figsize=(8, 6))
    plt.plot(block_sizes, np.sqrt(Ek_varianzas), 'o-', label='E_k')
    plt.plot(block_sizes, np.sqrt(Ep_varianzas), 'o-', label='E_p')
    plt.plot(block_sizes, np.sqrt(Et_varianzas), 'o-', label='E_t')
    plt.plot(block_sizes, np.sqrt(T_varianzas), 'o-', label='T')
    plt.plot(block_sizes, np.sqrt(P_varianzas), 'o-', label='P')
    plt.xlabel('Tamaño de bloque')
    plt.ylabel('Error estándar (sqrt(varianza))')
    plt.title('Error vs Tamaño de Bloque')
    plt.legend()
    plt.grid()
    plt.savefig(f"error_vs_block_size_{name}.png")
    plt.close()

    # Trayectorias y Velocidades
    plot_trayectorias(particulas_pos, L_adi, f"trayectorias_6_particulas_{name}.png")
    plot_velocidades(particulas_vel, tiempo, f"velocidades_6_particulas_{name}.png")

    # Distribución de partículas
    instante_titles = [
        "Distribución inicial (t = 0)",
        f"Distribución a t = Tc/4 (paso {steps//4})",
        f"Distribución a t = Tc/2 (paso {steps//2})",
        f"Distribución a t = Tc (paso {steps - 1})"
    ]
    if len(instantes) >= 4:
        interval = max(1, len(instantes) // 4)
        instante = instantes[::interval][:4]
        for i, (pos_inst, title) in enumerate(zip(instante, instante_titles)):
            plot_distribucion(pos_inst, L_adi, title, f"distribucion_paso_{i * interval * intervalo}_{name}.png")

    # Animación 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [], s=10)
    ax.set_xlim(-L_adi/2, L_adi/2)
    ax.set_ylim(-L_adi/2, L_adi/2)
    ax.set_zlim(-L_adi/2, L_adi/2)
    def actualizar(frame):
        pos = instantes[frame]
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return scatter,
    ani = FuncAnimation(fig, actualizar, frames=len(instantes), interval=int(2e3), blit=True)
    ani.save(f"animacion_3d_{name}.gif", writer='ffmpeg', fps=3)
    plt.close()

### ============================= SIMULACIÓN PRINCIPAL (N ≈ 1004, dt = 10^-14 s) ============================= ###
pos = init_posiciones(N, L_adi)
vel = init_velocidades(N, T0, m)
F, U, virial = calcular_fuerzas(pos, L_adi)

steps = 1500
Ek_list, Ep_list, Et_list, T_list, P_list, vcm_list = [], [], [], [], [], []
instantes = []
particulas_seleccionadas = np.random.choice(N, 6, replace=False)
particulas_pos = [[] for _ in range(6)]
particulas_vel = [[] for _ in range(6)]

print(f"Simulación principal: N = {N}, L = {L:.2e} m, L* = {L_adi:.2f}, dt = {dt:.1e} s, dt* = {dt_adi:.1e}, pasos totales = {steps}")

important_steps = [0, steps//4, steps//2, steps - 1]
safe_steps = [s - s % intervalo for s in important_steps]

for step in range(steps):
    pos, vel, F, U, virial = verlet(pos, vel, F, dt_adi, L_adi)
    
    if step % intervalo == 0:
        Ek = energia_cinetica(vel, m, N)
        T = temperatura(vel, m, N)
        Ep = U * epsilon / N
        Et = Ek + Ep
        P = presion(T, virial, V, N)
        vcm = np.linalg.norm(velocidad_cm(vel))
        Ek_list.append(Ek)
        Ep_list.append(Ep)
        Et_list.append(Et)
        T_list.append(T)
        P_list.append(P)
        vcm_list.append(vcm)
        
        for i, idx in enumerate(particulas_seleccionadas):
            particulas_pos[i].append(pos[idx].copy())
            particulas_vel[i].append(vel[idx].copy() * np.sqrt(epsilon / m) / sigma)
        
        if step in safe_steps:
            instantes.append(pos.copy())

particulas_pos = [np.array(pos_list) for pos_list in particulas_pos]
particulas_vel = [np.array(vel_list) for vel_list in particulas_vel]

df = pd.DataFrame({
    "E_k": Ek_list, "E_p": Ep_list, "E_t": Et_list, "T": T_list, "P": P_list, "v_cm": vcm_list
})
df.to_excel("resultados_N1004.xlsx", index=False)

block_sizes = [1, 2, 5, 10, 20]
Ek_varianzas = block_analisis(df['E_k'], block_sizes)
Ep_varianzas = block_analisis(df['E_p'], block_sizes)
Et_varianzas = block_analisis(df['E_t'], block_sizes)
T_varianzas = block_analisis(df['T'], block_sizes)
P_varianzas = block_analisis(df['P'], block_sizes)

block_size = 2
Ek_mean = np.mean(df['E_k'])
Ek_err = np.sqrt(Ek_varianzas[block_sizes.index(block_size)])
Ep_mean = np.mean(df['E_p'])
Ep_err = np.sqrt(Ep_varianzas[block_sizes.index(block_size)])
Et_mean = np.mean(df['E_t'])
Et_err = np.sqrt(Et_varianzas[block_sizes.index(block_size)])
T_mean = np.mean(df['T'])
T_err = np.sqrt(T_varianzas[block_sizes.index(block_size)])
P_mean = np.mean(df['P'])
P_err = np.sqrt(P_varianzas[block_sizes.index(block_size)])

Et_fluctuacion = np.std(df['E_t']) / abs(np.mean(df['E_t']))
print(f"Fluctuación relativa de la energía total: {Et_fluctuacion:.2e} (debería ser pequeña, e.g., < 0.01)")

df1 = pd.DataFrame({
    "Ek_mean": [Ek_mean], "Ek_err": [Ek_err], "Ek_varianza": [Ek_varianzas[block_sizes.index(block_size)]],
    "Ep_mean": [Ep_mean], "Ep_err": [Ep_err], "Ep_varianza": [Ep_varianzas[block_sizes.index(block_size)]],
    "Et_mean": [Et_mean], "Et_err": [Et_err], "Et_varianza": [Et_varianzas[block_sizes.index(block_size)]],
    "T_mean": [T_mean], "T_err": [T_err], "T_varianza": [T_varianzas[block_sizes.index(block_size)]],
    "P_mean": [P_mean], "P_err": [P_err], "P_varianza": [P_varianzas[block_sizes.index(block_size)]]
})
df1.to_excel("Medias_y_varianzas_N1004.xlsx", index=False)

print(f"Energía cinética media: {Ek_mean:.2e} ± {Ek_err:.2e} J/partícula")
print(f"Energía potencial media: {Ep_mean:.2e} ± {Ep_err:.2e} J/partícula")
print(f"Energía total media: {Et_mean:.2e} ± {Et_err:.2e} J/partícula")
print(f"Temperatura media: {T_mean:.2e} ± {T_err:.2e} K")
print(f"Presión media: {P_mean:.2e} ± {P_err:.2e} Pa")

tiempo = np.arange(len(Ek_list)) * intervalo * dt * 1e12
generar_graficas(Ek_list, Ep_list, Et_list, T_list, P_list, vcm_list, particulas_pos, particulas_vel, instantes, tiempo, L_adi, steps, block_sizes, "N1004")

### ============================= SIMULACIÓN CON N = 27 ============================= ###
N_27 = 27
L_27 = ((N_27 * m) / rho) ** (1 / 3)
L_adi_27 = L_27 / sigma
rc_27 = L_adi_27 / 2
rc_2_27 = rc_27 ** 2
V_27 = (L_adi_27 * sigma) ** 3
dt_adi_27 = dt * np.sqrt(epsilon / (m * sigma ** 2))

pos_27 = init_posiciones(N_27, L_adi_27)
vel_27 = init_velocidades(N_27, T0, m)
F_27, U_27, virial_27 = calcular_fuerzas(pos_27, L_adi_27)

steps_27 = 500
Ek_list_27, Ep_list_27, Et_list_27, T_list_27, P_list_27, vcm_list_27 = [], [], [], [], [], []
instantes_27 = []
particulas_seleccionadas_27 = np.random.choice(N_27, min(6, N_27), replace=False)
particulas_pos_27 = [[] for _ in range(len(particulas_seleccionadas_27))]
particulas_vel_27 = [[] for _ in range(len(particulas_seleccionadas_27))]

print('')
print('')
print(f"Simulación N=27: N = {N_27}, L = {L_27:.2e} m, L* = {L_adi_27:.2f}, dt = {dt:.1e} s, dt* = {dt_adi_27:.1e}, pasos totales = {steps_27}")

important_steps_27 = [0, steps_27//4, steps_27//2, steps_27 - 1]
safe_steps_27 = [s - s % intervalo for s in important_steps_27]

for step in range(steps_27):
    pos_27, vel_27, F_27, U_27, virial_27 = verlet(pos_27, vel_27, F_27, dt_adi_27, L_adi_27)
    if step % intervalo == 0:
        Ek = energia_cinetica(vel_27, m, N_27)
        T = temperatura(vel_27, m, N_27)
        Ep = U_27 * epsilon / N_27
        Et = Ek + Ep
        P = presion(T, virial_27, V_27, N_27)
        vcm = np.linalg.norm(velocidad_cm(vel_27))
        Ek_list_27.append(Ek)
        Ep_list_27.append(Ep)
        Et_list_27.append(Et)
        T_list_27.append(T)
        P_list_27.append(P)
        vcm_list_27.append(vcm)
        
        for i, idx in enumerate(particulas_seleccionadas_27):
            particulas_pos_27[i].append(pos_27[idx].copy())
            particulas_vel_27[i].append(vel_27[idx].copy() * np.sqrt(epsilon / m) / sigma)
        
        if step in safe_steps_27:
            instantes_27.append(pos_27.copy())

particulas_pos_27 = [np.array(pos_list) for pos_list in particulas_pos_27]
particulas_vel_27 = [np.array(vel_list) for vel_list in particulas_vel_27]

df_27 = pd.DataFrame({
    "E_k": Ek_list_27, "E_p": Ep_list_27, "E_t": Et_list_27, "T": T_list_27, "P": P_list_27, "v_cm": vcm_list_27
})
df_27.to_excel("resultados_N27.xlsx", index=False)

Ek_mean_27 = np.mean(Ek_list_27)
Ep_mean_27 = np.mean(Ep_list_27)
Et_mean_27 = np.mean(Ek_list_27)
T_mean_27 = np.mean(T_list_27)
P_mean_27 = np.mean(P_list_27)

block_sizes = [1, 2, 5, 10, 20]
Ek_varianzas = block_analisis(df['E_k'], block_sizes)
Ep_varianzas = block_analisis(df['E_p'], block_sizes)
Et_varianzas = block_analisis(df['E_t'], block_sizes)
T_varianzas = block_analisis(df['T'], block_sizes)
P_varianzas = block_analisis(df['P'], block_sizes)

block_size = 2
Ek_mean = np.mean(df['E_k'])
Ek_err = np.sqrt(Ek_varianzas[block_sizes.index(block_size)])
Ep_mean = np.mean(df['E_p'])
Ep_err = np.sqrt(Ep_varianzas[block_sizes.index(block_size)])
Et_mean = np.mean(df['E_t'])
Et_err = np.sqrt(Et_varianzas[block_sizes.index(block_size)])
T_mean = np.mean(df['T'])
T_err = np.sqrt(T_varianzas[block_sizes.index(block_size)])
P_mean = np.mean(df['P'])
P_err = np.sqrt(P_varianzas[block_sizes.index(block_size)])

df2 = pd.DataFrame({
    "Ek_mean": [Ek_mean], "Ek_err": [Ek_err], "Ek_varianza": [Ek_varianzas[block_sizes.index(block_size)]],
    "Ep_mean": [Ep_mean], "Ep_err": [Ep_err], "Ep_varianza": [Ep_varianzas[block_sizes.index(block_size)]],
    "Et_mean": [Et_mean], "Et_err": [Et_err], "Et_varianza": [Et_varianzas[block_sizes.index(block_size)]],
    "T_mean": [T_mean], "T_err": [T_err], "T_varianza": [T_varianzas[block_sizes.index(block_size)]],
    "P_mean": [P_mean], "P_err": [P_err], "P_varianza": [P_varianzas[block_sizes.index(block_size)]]
})
df2.to_excel("Medias_y_varianzas_N27.xlsx", index=False)

print(f"Energía cinética media: {Ek_mean:.2e} ± {Ek_err:.2e} J/partícula")
print(f"Energía potencial media: {Ep_mean:.2e} ± {Ep_err:.2e} J/partícula")
print(f"Energía total media: {Et_mean:.2e} ± {Et_err:.2e} J/partícula")
print(f"Temperatura media: {T_mean:.2e} ± {T_err:.2e} K")
print(f"Presión media: {P_mean:.2e} ± {P_err:.2e} Pa")

Et_fluctuacion = np.std(df['E_t']) / abs(np.mean(df['E_t']))
print(f"Fluctuación relativa de la energía total: {Et_fluctuacion:.2e} (debería ser pequeña, e.g., < 0.01)")

print(f"Resultados N=27: E_k = {Ek_mean_27:.2e}, E_p = {Ep_mean_27:.2e}, E_t = {Et_mean_27:.2e}, T = {T_mean_27:.2e}, P = {P_mean_27:.2e}")

tiempo_27 = np.arange(len(Ek_list_27)) * intervalo * dt * 1e12
generar_graficas(Ek_list_27, Ep_list_27, Et_list_27, T_list_27, P_list_27, vcm_list_27, particulas_pos_27, particulas_vel_27, instantes_27, tiempo_27, L_adi_27, steps_27, block_sizes, "N27")

### ============================= SIMULACIÓN CON dt = 10^-9 s ============================= ###
dt_large = 1e-9
dt_adi_large = dt_large * np.sqrt(epsilon / (m * sigma ** 2))

pos_dt = init_posiciones(N, L_adi)
vel_dt = init_velocidades(N, T0, m)
F_dt, U_dt, virial_dt = calcular_fuerzas(pos_dt, L_adi)

steps_dt = 500
Ek_list_dt, Ep_list_dt, Et_list_dt, T_list_dt, P_list_dt, vcm_list_dt = [], [], [], [], [], []
instantes_dt = []
particulas_seleccionadas_dt = np.random.choice(N, 6, replace=False)
particulas_pos_dt = [[] for _ in range(6)]
particulas_vel_dt = [[] for _ in range(6)]

print('')
print('')
print(f"Simulación dt=10^-9 s: N = {N}, L = {L:.2e} m, L* = {L_adi:.2f}, dt = {dt_large:.1e} s, dt* = {dt_adi_large:.1e}, pasos totales = {steps_dt}")

important_steps_dt = [0, steps_dt//4, steps_dt//2, steps_dt - 1]
safe_steps_dt = [s - s % intervalo for s in important_steps_dt]

for step in range(steps_dt):
    pos_dt, vel_dt, F_dt, U_dt, virial_dt = verlet(pos_dt, vel_dt, F_dt, dt_adi_large, L_adi)
    if step % intervalo == 0:
        Ek = energia_cinetica(vel_dt, m, N)
        T = temperatura(vel_dt, m, N)
        Ep = U_dt * epsilon / N
        Et = Ek + Ep
        P = presion(T, virial_dt, V, N)
        vcm = np.linalg.norm(velocidad_cm(vel_dt))
        Ek_list_dt.append(Ek)
        Ep_list_dt.append(Ep)
        Et_list_dt.append(Et)
        T_list_dt.append(T)
        P_list_dt.append(P)
        vcm_list_dt.append(vcm)
        
        for i, idx in enumerate(particulas_seleccionadas_dt):
            particulas_pos_dt[i].append(pos_dt[idx].copy())
            particulas_vel_dt[i].append(vel_dt[idx].copy() * np.sqrt(epsilon / m) / sigma)
        
        if step in safe_steps_dt:
            instantes_dt.append(pos_dt.copy())

particulas_pos_dt = [np.array(pos_list) for pos_list in particulas_pos_dt]
particulas_vel_dt = [np.array(vel_list) for vel_list in particulas_vel_dt]

df_dt = pd.DataFrame({
    "E_k": Ek_list_dt, "E_p": Ep_list_dt, "E_t": Et_list_dt, "T": T_list_dt, "P": P_list_dt, "v_cm": vcm_list_dt
})
df_dt.to_excel("resultados_dt_large.xlsx", index=False)

Ek_mean_dt = np.mean(Ek_list_dt)
Ep_mean_dt = np.mean(Ep_list_dt)
Et_mean_dt = np.mean(Ek_list_dt)
T_mean_dt = np.mean(T_list_dt)
P_mean_dt = np.mean(P_list_dt)


block_sizes = [1, 2, 5, 10, 20]
Ek_varianzas = block_analisis(df['E_k'], block_sizes)
Ep_varianzas = block_analisis(df['E_p'], block_sizes)
Et_varianzas = block_analisis(df['E_t'], block_sizes)
T_varianzas = block_analisis(df['T'], block_sizes)
P_varianzas = block_analisis(df['P'], block_sizes)

block_size = 2
Ek_mean = np.mean(df['E_k'])
Ek_err = np.sqrt(Ek_varianzas[block_sizes.index(block_size)])
Ep_mean = np.mean(df['E_p'])
Ep_err = np.sqrt(Ep_varianzas[block_sizes.index(block_size)])
Et_mean = np.mean(df['E_t'])
Et_err = np.sqrt(Et_varianzas[block_sizes.index(block_size)])
T_mean = np.mean(df['T'])
T_err = np.sqrt(T_varianzas[block_sizes.index(block_size)])
P_mean = np.mean(df['P'])
P_err = np.sqrt(P_varianzas[block_sizes.index(block_size)])

df3 = pd.DataFrame({
    "Ek_mean": [Ek_mean], "Ek_err": [Ek_err], "Ek_varianza": [Ek_varianzas[block_sizes.index(block_size)]],
    "Ep_mean": [Ep_mean], "Ep_err": [Ep_err], "Ep_varianza": [Ep_varianzas[block_sizes.index(block_size)]],
    "Et_mean": [Et_mean], "Et_err": [Et_err], "Et_varianza": [Et_varianzas[block_sizes.index(block_size)]],
    "T_mean": [T_mean], "T_err": [T_err], "T_varianza": [T_varianzas[block_sizes.index(block_size)]],
    "P_mean": [P_mean], "P_err": [P_err], "P_varianza": [P_varianzas[block_sizes.index(block_size)]]
})
df3.to_excel("Medias_y_varianzas_dt_large.xlsx", index=False)

print(f"Energía cinética media: {Ek_mean:.2e} ± {Ek_err:.2e} J/partícula")
print(f"Energía potencial media: {Ep_mean:.2e} ± {Ep_err:.2e} J/partícula")
print(f"Energía total media: {Et_mean:.2e} ± {Et_err:.2e} J/partícula")
print(f"Temperatura media: {T_mean:.2e} ± {T_err:.2e} K")
print(f"Presión media: {P_mean:.2e} ± {P_err:.2e} Pa")

Et_fluctuacion_dt = np.std(df_dt['E_t']) / abs(np.mean(df_dt['E_t']))
print(f"Fluctuación relativa de la energía total (dt=10^-9 s): {Et_fluctuacion_dt:.2e}")

print(f"Resultados dt=10^-9 s: E_k = {Ek_mean_dt:.2e}, E_p = {Ep_mean_dt:.2e}, E_t = {Et_mean_dt:.2e}, T = {T_mean_dt:.2e}, P = {P_mean_dt:.2e}")

tiempo_dt = np.arange(len(Ek_list_dt)) * intervalo * dt_large * 1e12
generar_graficas(Ek_list_dt, Ep_list_dt, Et_list_dt, T_list_dt, P_list_dt, vcm_list_dt, particulas_pos_dt, particulas_vel_dt, instantes_dt, tiempo_dt, L_adi, steps_dt, block_sizes, "dt_large")