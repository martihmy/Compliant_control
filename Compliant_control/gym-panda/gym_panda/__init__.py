from gym.envs.registration import register

register(
    id = 'panda-v0',
    entry_points = 'gym_panda.envs:PandaEnv',
)