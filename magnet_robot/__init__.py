from gym.envs.registration import register

register(
    id='magnet_robot-v0',
    entry_point='magnet_robot.envs:Magnet_robot',
)