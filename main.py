import numpy as np
import torch
from agents.teacher import TeachingAgent
from agents.student import StudentAgent
from environments.task_env import TeachingTaskEnv
import matplotlib.pyplot as plt

def train_agents(num_episodes=1000, teacher_learning_rate=0.001, student_learning_rate=0.001):
    # Initialize environment
    env = TeachingTaskEnv(task_complexity=0.5)
    
    # Initialize agents
    teacher = TeachingAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        curriculum_dim=env.state_dim,
        learning_rate=teacher_learning_rate
    )
    
    student = StudentAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=student_learning_rate
    )
    
    # Training metrics
    teacher_rewards = []
    student_rewards = []
    curriculum_difficulties = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_teacher_reward = 0
        episode_student_reward = 0
        
        # Teacher's turn
        teacher_action = teacher.select_action(state)
        curriculum = teacher.generate_curriculum(state, teacher_action)
        
        # Update environment difficulty based on curriculum
        env.set_task_difficulty(np.mean(curriculum))
        
        # Student's turn
        student_understanding = student.understand_curriculum(state, curriculum)
        student_action = student.select_action(student_understanding)
        
        # Environment step
        next_state, reward, done, _, info = env.step(student_action)
        
        # Update agents
        teacher_experience = (state, teacher_action, reward, next_state)
        student_experience = (state, student_action, reward, next_state, curriculum)
        
        teacher_metrics = teacher.update(teacher_experience)
        student_metrics = student.update(student_experience)
        
        # Record metrics
        episode_teacher_reward += reward
        episode_student_reward += reward
        curriculum_difficulties.append(env.get_task_difficulty())
        
        if done:
            teacher_rewards.append(episode_teacher_reward)
            student_rewards.append(episode_student_reward)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}")
                print(f"Teacher Reward: {np.mean(teacher_rewards[-100:]):.2f}")
                print(f"Student Reward: {np.mean(student_rewards[-100:]):.2f}")
                print(f"Curriculum Difficulty: {np.mean(curriculum_difficulties[-100:]):.2f}")
                print("---")
    
    return teacher_rewards, student_rewards, curriculum_difficulties

def plot_results(teacher_rewards, student_rewards, curriculum_difficulties):
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(teacher_rewards, label='Teacher')
    plt.plot(student_rewards, label='Student')
    plt.title('Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot curriculum difficulty
    plt.subplot(1, 2, 2)
    plt.plot(curriculum_difficulties)
    plt.title('Curriculum Difficulty over Time')
    plt.xlabel('Episode')
    plt.ylabel('Difficulty')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train agents
    teacher_rewards, student_rewards, curriculum_difficulties = train_agents()
    
    # Plot results
    plot_results(teacher_rewards, student_rewards, curriculum_difficulties) 