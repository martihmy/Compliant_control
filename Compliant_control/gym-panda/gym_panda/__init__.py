from gym.envs.registration import register

register(
    id = 'panda-admittance-v0',
    entry_point = 'gym_panda.envs:AdmittanceEnv',
)

register(
    id = 'panda-HMFC-v0',
    entry_point = 'gym_panda.envs:HMFC_Env',
)

register(
    id = 'panda-VIC-v0',
    entry_point = 'gym_panda.envs:VIC_Env',
)

