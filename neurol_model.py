from brian2 import *
import numpy as np
def simulate_hh_ei_lfp(
    N=50,
    frac_exc=0.8,
    p_conn=0.1,
    w_e=0.2*msiemens,
    w_i=0.5*msiemens,
    ext_rate=5*Hz,
    ext_w=0.1*msiemens,
    I_inj_value=0.5*uA,
    prewarm=100*ms,
    duration=500*ms,
    Ne=3,
    neuron_spacing=100*um,
    electrode_base_dist=200*um,
    dt=0.01*ms,
    do_plot=false,
    rng_seed=None,
):
    """
    使用 HH 点神经元 + E/I 电导突触 + 多电极 LFP（1/r 衰减）的网络仿真。

    返回:
        t          : 时间轴 (ms), shape (T,)
        lfp_array  : LFP 数组 (mV), shape (Ne, T)
        params     : 本次仿真用到的参数 dict（N, frac_exc, p_conn, seed 等）
    """

    # ===== 随机种子控制 =====
    if rng_seed is not None:
        np.random.seed(int(rng_seed))
        seed(int(rng_seed))   # Brian2 自己的随机数种子

    start_scope()
    defaultclock.dt = dt

    # ==========================
    # 1. HH 神经元参数（点神经元）
    # ==========================
    # 这里沿用你之前的参数，只是理解为“每个神经元等效面积 ~ 1 cm^2”
    Cm  = 1*uF/cm**2
    Cm  = Cm * (1*cm**2)        # F
    El  = 10.613*mV
    ENa = 115*mV
    EK  = -12*mV
    gl  = 0.3*msiemens/cm**2 * (1*cm**2)   # -> siemens
    gNa = 120*msiemens/cm**2 * (1*cm**2)   # -> siemens
    gK  = 36*msiemens/cm**2 * (1*cm**2)    # -> siemens

    # 突触参数
    E_exc = 0*mV
    E_inh = -80*mV
    tau_e = 5*ms
    tau_i = 10*ms

    # ==========================
    # 2. 多神经元 HH 方程（点神经元）+ 位置
    # ==========================
    eqs = '''
    Im = gl * (El-v) + gNa * m**3 * h * (ENa-v) + gK * n**4 * (EK-v)
         + ge * (E_exc-v) + gi * (E_inh-v) + I_inj : amp

    dv/dt = Im / Cm : volt

    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens

    I_inj : amp   # 外加电流

    dm/dt = alpham * (1-m) - betam * m : 1
    dn/dt = alphan * (1-n) - betan * n : 1
    dh/dt = alphah * (1-h) - betah * h : 1

    alpham = (0.1/mV) * 10*mV/exprel((-v+25*mV)/(10*mV))/ms : Hz
    betam = 4 * exp(-v/(18*mV))/ms : Hz
    alphah = 0.07 * exp(-v/(20*mV))/ms : Hz
    betah = 1/(exp((-v+30*mV) / (10*mV)) + 1)/ms : Hz
    alphan = (0.01/mV) * 10*mV/exprel((-v+10*mV)/(10*mV))/ms : Hz
    betan = 0.125*exp(-v/(80*mV))/ms : Hz

    x : meter
    y : meter
    z : meter
    '''

    # ==========================
    # 3. 建一个 E/I 群体：多 HH 神经元
    # ==========================
    N_exc = int(N*frac_exc)
    N_inh = N - N_exc

    G = NeuronGroup(N, eqs,
                    threshold='v > 0*mV',
                    refractory='v > -40*mV',
                    method='exponential_euler')

    # 初始化
    G.v = El
    G.m = 0
    G.h = 1
    G.n = 0
    G.ge = 0*siemens
    G.gi = 0*siemens
    G.I_inj = 0*amp

    # 简单放在一条线上：x 方向等间距，y=z=0
    G.x = 'i * neuron_spacing'
    G.y = 0*um
    G.z = 0*um

    Ge = G[:N_exc]
    Gi = G[N_exc:]

    # ==========================
    # 4. E/I 突触（电导型）
    # ==========================
    S_ee = Synapses(Ge, G, on_pre='ge_post += w_e')
    S_ee.connect(p=p_conn)

    S_ie = Synapses(Gi, G, on_pre='gi_post += w_i')
    S_ie.connect(p=p_conn)

    # 背景 Poisson 兴奋输入
    if N_exc > 0:
        P = PoissonGroup(N_exc, rates=ext_rate)
        S_ext = Synapses(P, Ge, on_pre='ge_post += ext_w')
        S_ext.connect(j='i')

    # 给一部分 E 神经元一点持续注流，启动网络
    n_inj = min(10, N_exc)
    if n_inj > 0:
        G.I_inj[:n_inj] = I_inj_value

    # ==========================
    # 5. LFP 记录（多电极 + 距离衰减 1/r）
    # ==========================
    sigma = 0.3*siemens/meter       # 组织电导率
    rho = 1/sigma                   # 电阻率 (ohm*meter)

    lfp = NeuronGroup(Ne, '''
    v : volt
    x : meter
    y : meter
    z : meter
    ''')
    lfp.v = 0*mV

    # 电极放在网络中间，y 方向距离逐渐增大
    center_x = (N-1)/2 * neuron_spacing
    lfp.x = center_x
    # y_i = (i+1) * electrode_base_dist
    lfp.y = '(i+1)*electrode_base_dist'
    lfp.z = 0*um

    S_lfp = Synapses(G, lfp,
                     model='''
                     w : ohm (constant)
                     v_post = w * Im_pre : volt (summed)
                     ''')

    S_lfp.connect()

    # w = rho / (4*pi*r)  （r为神经元到电极的距离），单位 ohm
    S_lfp.w = 'rho / (4*pi*sqrt((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2))'

    # ==========================
    # 6. 预热 + 正式仿真
    # ==========================
    # 预热，让状态先收敛一会儿
    run(prewarm)

    # 清零 LFP，重新开监视器
    lfp.v = 0*mV
    M_lfp = StateMonitor(lfp, 'v', record=True)
    M_v = StateMonitor(G, 'v', record=0)
    spikes = SpikeMonitor(G)

    run(duration)

    # ==========================
    # 7. 整理输出 + 可视化（可选）
    # ==========================
    t = M_lfp.t/ms                   # 时间轴 (ms)
    lfp_array = np.array(M_lfp.v/mV) # shape: [Ne, T]

    params = dict(
        N=int(N),
        frac_exc=float(frac_exc),
        p_conn=float(p_conn),
        w_e=float(w_e/(msiemens)),
        w_i=float(w_i/(msiemens)),
        ext_rate_Hz=float(ext_rate/Hz),
        ext_w=float(ext_w/(msiemens)),
        I_inj_value_nA=float(I_inj_value/nA),
        prewarm_ms=float(prewarm/ms),
        duration_ms=float(duration/ms),
        Ne=int(Ne),
        neuron_spacing_um=float(neuron_spacing/um),
        electrode_base_dist_um=float(electrode_base_dist/um),
        dt_ms=float(dt/ms),
        seed=int(rng_seed) if rng_seed is not None else -1,
    )

    if do_plot:
        figure(figsize=(8,6))
        subplot(3,1,1)
        plot(M_v.t/ms, M_v.v[0]/mV)
        ylabel('V_0 (mV)')
        title('Neuron 0 membrane potential')

        subplot(3,1,2)
        plot(spikes.t/ms, spikes.i, '.k', markersize=3)
        ylabel('Neuron index')
        title('Raster')

        subplot(3,1,3)
        for i in range(Ne):
            plot(M_lfp.t/ms, M_lfp.v[i]/mV, label=f'LFP {i}')
        ylabel('LFP (mV)')
        xlabel('Time (ms)')
        legend()
        tight_layout()
        show()

    return t, lfp_array, params

if __name__ == "__main__":
    # ====== 你要扫的参数网格 ======
    N_list = [50, 100, 200, 500, 1000, 5000, 10000]   # 神经元数量
    p_conn_list = [0.1, 0.3, 0.5]                     # 连接率
    frac_exc_list = [0.4, 0.6, 0.8]                   # 兴奋比例

    n_repeats = 3   # ★ 每组参数重复次数，你可以改大/改小

    duration = 2*second
    prewarm = 500*ms
    dt = 0.01*ms
    Ne = 3

    all_lfps = []
    all_params = []

    # 计算总仿真次数
    total_sims = len(N_list) * len(p_conn_list) * len(frac_exc_list) * n_repeats
    sim_idx = 0

    for N in N_list:
        for p_conn in p_conn_list:
            for frac_exc in frac_exc_list:
                for rep in range(n_repeats):
                    sim_idx += 1
                    # 生成一个简单的随机种子（你也可以用别的方式）
                    rng_seed = sim_idx

                    print(
                        f"[{sim_idx}/{total_sims}] "
                        f"N={N}, p_conn={p_conn}, frac_exc={frac_exc}, "
                        f"rep={rep+1}/{n_repeats}, seed={rng_seed}"
                    )

                    t, lfp_array, params = simulate_hh_ei_lfp(
                        N=N,
                        frac_exc=frac_exc,
                        p_conn=p_conn,
                        duration=duration,
                        prewarm=prewarm,
                        dt=dt,
                        Ne=Ne,
                        rng_seed=rng_seed,   # ★ 把 seed 传进去
                    )

                    # 把 seed 也存到 params 里（以防你在 simulate 里忘记加）
                    params["seed"] = rng_seed

                    all_lfps.append(lfp_array)   # [Ne, T]
                    all_params.append(params)

    # 把 list 堆成 numpy 数组
    all_lfps = np.stack(all_lfps, axis=0)  # 形状: [num_sims, Ne, T]

    # 把参数拆开成几个数组方便标注
    N_arr = np.array([p["N"] for p in all_params], dtype=int)
    p_conn_arr = np.array([p["p_conn"] for p in all_params], dtype=float)
    frac_exc_arr = np.array([p["frac_exc"] for p in all_params], dtype=float)
    seed_arr = np.array([p.get("seed", -1) for p in all_params], dtype=int)

    # 保存成 npz
    out_fname = "lfp_dataset.npz"
    np.savez(
        out_fname,
        t=t,                          # 时间轴 (ms)
        lfps=all_lfps,                # [num_sims, Ne, T]
        N=N_arr,
        p_conn=p_conn_arr,
        frac_exc=frac_exc_arr,
        seed=seed_arr,
        dt_ms=float(dt/ms),
        duration_ms=float(duration/ms),
    )

    print(f"Saved {all_lfps.shape[0]} LFP samples to {out_fname}")
    print(f"LFP shape = {all_lfps.shape}, time steps = {all_lfps.shape[-1]}")

