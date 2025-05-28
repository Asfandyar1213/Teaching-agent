import numpy as np
import torch
import argparse
import logging
import os
import json
import datetime
from pathlib import Path
from agents.teacher import TeachingAgent
from agents.student import StudentAgent
from environments.task_env import TeachingTaskEnv
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directories
LOGS_DIR = Path('logs')
CHECKPOINTS_DIR = Path('checkpoints')
CONFIGS_DIR = Path('configs')
RESULTS_DIR = Path('results')

for dir_path in [LOGS_DIR, CHECKPOINTS_DIR, CONFIGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

def setup_logging(log_level='INFO', log_file=None):
    """Set up logging with specified level and file."""
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOGS_DIR / f'experiment_{timestamp}.log'
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        filename=str(log_file),
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=numeric_level
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    logging.getLogger('').addHandler(console)
    
    return logging.getLogger(__name__)

def save_checkpoint(agent, name, episode=None):
    """Save agent checkpoint with optional episode number."""
    if episode is not None:
        path = CHECKPOINTS_DIR / f'{name}_episode_{episode}.pth'
    else:
        path = CHECKPOINTS_DIR / f'{name}.pth'
    agent.save_model(str(path))
    logger.info(f"Checkpoint saved: {path}")

def load_checkpoint(agent, name, episode=None):
    """Load agent checkpoint with optional episode number."""
    if episode is not None:
        path = CHECKPOINTS_DIR / f'{name}_episode_{episode}.pth'
    else:
        path = CHECKPOINTS_DIR / f'{name}.pth'
    
    if path.exists():
        agent.load_model(str(path))
        logger.info(f"Checkpoint loaded: {path}")
        return True
    else:
        logger.warning(f"Checkpoint not found: {path}")
        return False

def save_config(config, name):
    """Save configuration to file."""
    config_path = CONFIGS_DIR / f'{name}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved: {config_path}")

def load_config(name):
    """Load configuration from file."""
    config_path = CONFIGS_DIR / f'{name}.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def train_agents(config):
    """Train agents with configuration dictionary."""
    # Initialize environment
    env = TeachingTaskEnv(task_complexity=config['env_difficulty'])
    
    # Initialize agents
    teacher = TeachingAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        curriculum_dim=env.state_dim,
        learning_rate=config['teacher_learning_rate']
    )
    
    student = StudentAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=config['student_learning_rate']
    )
    
    if config['resume']:
        load_checkpoint(teacher, 'teacher', config.get('resume_episode'))
        load_checkpoint(student, 'student', config.get('resume_episode'))
    
    # Training metrics
    metrics = {
        'teacher_rewards': [],
        'student_rewards': [],
        'curriculum_difficulties': [],
        'teacher_losses': [],
        'student_losses': []
    }
    
    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        episode_teacher_reward = 0
        episode_student_reward = 0
        
        # Teacher's turn
        teacher_action = teacher.select_action(state)
        curriculum = teacher.generate_curriculum(state, teacher_action)
        logger.info(f"Episode {episode+1}: Teacher action {teacher_action}, Curriculum {curriculum}")
        
        # Update environment difficulty based on curriculum
        env.set_task_difficulty(np.mean(curriculum))
        
        # Student's turn
        student_understanding = student.understand_curriculum(state, curriculum)
        student_action = student.select_action(student_understanding)
        logger.info(f"Episode {episode+1}: Student action {student_action}")
        
        # Environment step
        next_state, reward, done, _, info = env.step(student_action)
        logger.info(f"Episode {episode+1}: Reward {reward}, Done {done}, Info {info}")
        
        # Update agents
        teacher_experience = (state, teacher_action, reward, next_state)
        student_experience = (state, student_action, reward, next_state, curriculum)
        
        teacher_metrics = teacher.update(teacher_experience)
        student_metrics = student.update(student_experience)
        
        # Record metrics
        episode_teacher_reward += reward
        episode_student_reward += reward
        metrics['curriculum_difficulties'].append(env.get_task_difficulty())
        metrics['teacher_losses'].append(teacher_metrics.get('loss', 0))
        metrics['student_losses'].append(student_metrics.get('loss', 0))
        
        if done:
            metrics['teacher_rewards'].append(episode_teacher_reward)
            metrics['student_rewards'].append(episode_student_reward)
            
            if (episode + 1) % config['log_interval'] == 0:
                log_metrics(metrics, episode + 1, config['log_interval'])
        
        # Save checkpoints
        if (episode + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint(teacher, 'teacher', episode + 1)
            save_checkpoint(student, 'student', episode + 1)
    
    # Final checkpoint
    save_checkpoint(teacher, 'teacher', config['num_episodes'])
    save_checkpoint(student, 'student', config['num_episodes'])
    
    return metrics

def log_metrics(metrics, episode, window):
    """Log training metrics."""
    logger.info(f"Episode {episode}")
    logger.info(f"Teacher Reward: {np.mean(metrics['teacher_rewards'][-window:]):.2f}")
    logger.info(f"Student Reward: {np.mean(metrics['student_rewards'][-window:]):.2f}")
    logger.info(f"Curriculum Difficulty: {np.mean(metrics['curriculum_difficulties'][-window:]):.2f}")
    logger.info(f"Teacher Loss: {np.mean(metrics['teacher_losses'][-window:]):.4f}")
    logger.info(f"Student Loss: {np.mean(metrics['student_losses'][-window:]):.4f}")
    
    print(f"Episode {episode}")
    print(f"Teacher Reward: {np.mean(metrics['teacher_rewards'][-window:]):.2f}")
    print(f"Student Reward: {np.mean(metrics['student_rewards'][-window:]):.2f}")
    print(f"Curriculum Difficulty: {np.mean(metrics['curriculum_difficulties'][-window:]):.2f}")
    print(f"Teacher Loss: {np.mean(metrics['teacher_losses'][-window:]):.4f}")
    print(f"Student Loss: {np.mean(metrics['student_losses'][-window:]):.4f}")
    print("---")

def evaluate_agents(config):
    """Evaluate agents with configuration dictionary."""
    env = TeachingTaskEnv(task_complexity=config['env_difficulty'])
    teacher = TeachingAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        curriculum_dim=env.state_dim
    )
    student = StudentAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )
    
    load_checkpoint(teacher, 'teacher', config.get('checkpoint_episode'))
    load_checkpoint(student, 'student', config.get('checkpoint_episode'))
    
    metrics = {
        'teacher_rewards': [],
        'student_rewards': [],
        'curriculum_difficulties': []
    }
    
    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        teacher_action = teacher.select_action(state)
        curriculum = teacher.generate_curriculum(state, teacher_action)
        env.set_task_difficulty(np.mean(curriculum))
        student_understanding = student.understand_curriculum(state, curriculum)
        student_action = student.select_action(student_understanding)
        next_state, reward, done, _, info = env.step(student_action)
        
        metrics['teacher_rewards'].append(reward)
        metrics['student_rewards'].append(reward)
        metrics['curriculum_difficulties'].append(env.get_task_difficulty())
    
    print(f"Evaluation over {config['num_episodes']} episodes:")
    print(f"Avg Teacher Reward: {np.mean(metrics['teacher_rewards']):.2f}")
    print(f"Avg Student Reward: {np.mean(metrics['student_rewards']):.2f}")
    print(f"Avg Curriculum Difficulty: {np.mean(metrics['curriculum_difficulties']):.2f}")
    
    logger.info(f"Evaluation Results:")
    logger.info(f"Avg Teacher Reward: {np.mean(metrics['teacher_rewards']):.2f}")
    logger.info(f"Avg Student Reward: {np.mean(metrics['student_rewards']):.2f}")
    logger.info(f"Avg Curriculum Difficulty: {np.mean(metrics['curriculum_difficulties']):.2f}")
    
    return metrics

def plot_results(metrics, save_path=None):
    """Plot training results with enhanced visualizations."""
    if save_path is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_DIR / f'training_results_{timestamp}.png'
    
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['teacher_rewards'], label='Teacher', alpha=0.7)
    plt.plot(metrics['student_rewards'], label='Student', alpha=0.7)
    plt.title('Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot curriculum difficulty
    plt.subplot(2, 2, 2)
    plt.plot(metrics['curriculum_difficulties'], color='green', alpha=0.7)
    plt.title('Curriculum Difficulty over Time')
    plt.xlabel('Episode')
    plt.ylabel('Difficulty')
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(metrics['teacher_losses'], label='Teacher', alpha=0.7)
    plt.plot(metrics['student_losses'], label='Student', alpha=0.7)
    plt.title('Losses over Time')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot reward distributions
    plt.subplot(2, 2, 4)
    sns.kdeplot(metrics['teacher_rewards'], label='Teacher', alpha=0.7)
    sns.kdeplot(metrics['student_rewards'], label='Student', alpha=0.7)
    plt.title('Reward Distributions')
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Results saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Teaching Agents CLI")
    parser.add_argument('--mode', choices=['train', 'eval', 'plot'], default='train',
                      help='Mode to run: train, eval, plot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, help='Number of episodes')
    parser.add_argument('--env_difficulty', type=float, help='Initial environment difficulty (0.0-1.0)')
    parser.add_argument('--checkpoint_interval', type=int, help='Episodes between checkpoints')
    parser.add_argument('--log_interval', type=int, help='Episodes between logging')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--resume_episode', type=int, help='Episode to resume from')
    parser.add_argument('--checkpoint_episode', type=int, help='Episode to load for evaluation')
    parser.add_argument('--teacher_lr', type=float, help='Teacher learning rate')
    parser.add_argument('--student_lr', type=float, help='Student learning rate')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='Logging level')
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
        if config is None:
            logger.error(f"Configuration file not found: {args.config}")
            return
    
    # Update config with command line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    
    # Set default values
    default_config = {
        'num_episodes': 1000,
        'env_difficulty': 0.5,
        'checkpoint_interval': 100,
        'log_interval': 100,
        'teacher_learning_rate': 0.001,
        'student_learning_rate': 0.001,
        'resume': False
    }
    
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # Setup logging
    global logger
    logger = setup_logging(config.get('log_level', 'INFO'))
    
    if args.mode == 'train':
        metrics = train_agents(config)
        plot_results(metrics)
    elif args.mode == 'eval':
        metrics = evaluate_agents(config)
        plot_results(metrics)
    elif args.mode == 'plot':
        # For demonstration, plot last training results if available
        results_files = list(RESULTS_DIR.glob('training_results_*.png'))
        if results_files:
            latest_result = max(results_files, key=lambda x: x.stat().st_mtime)
            from PIL import Image
            img = Image.open(latest_result)
            img.show()
        else:
            print("No training results found. Run training first.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    main() 