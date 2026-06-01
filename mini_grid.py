agent_x = 0
agent_y = 0

food_x = 2
food_y = 0

for step in range(5):
    print("\nStep:", step)
    print("Agent:", agent_x, agent_y)
    print("Food:", food_x, food_y)

    command = input("Move (w,a,s,d): ")

    if command == "w":
        agent_y -= 1
    elif command == "s":
        agent_y += 1
    elif command == "a":
        agent_x -= 1
    elif command == "d":
        agent_x += 1

    reward = 0

    if agent_x == food_x and agent_y == food_y:
        reward = 1
        print("Food collected!")
        print("Reward:", reward)
        break

    print("Reward:", reward)

print("Episode finished")