from gymnasium.envs.registration import register

register(
     id="four_rooms/FourRooms-v0",
     entry_point="four_rooms.envs:FourRoomsEnv",
     max_episode_steps=1000,
)
