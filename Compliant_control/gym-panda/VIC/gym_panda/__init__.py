from gym.envs.registration import register


register(
    id = 'panda-VIC-v0',
    entry_point = 'gym_panda.envs:VIC_Env',
)

