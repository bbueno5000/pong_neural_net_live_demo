"""
#DQN. CNN reads in pixel data. 
#reinforcement learning. trial and error.
#maximize action based on reward
#agent envrioment loop
#this is called Q Learning
#based on just game state. mapping of state to action is policy
#experience replay. learns from past policies
"""
import collections
import cv2
import numpy
import pygame
import random
import tensorflow

BALL_HEIGHT = 10
BALL_WIDTH = 10
BALL_X_SPEED = 3
BALL_Y_SPEED = 2
BLACK = (0, 0, 0)
PADDLE_BUFFER = 10
PADDLE_HEIGHT = 60
PADDLE_SPEED = 2
PADDLE_WIDTH = 10
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

class Pong:
    """
    DOCSTRING
    """
    def draw_ball(ballXPos, ballYPos):
        """
        DOCSTRING
        """
        ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
        pygame.draw.rect(screen, WHITE, ball)

    def draw_paddle_1(paddle1YPos):
        """
        DOCSTRING
        """
        paddle1 = pygame.Rect(PADDLE_BUFFER, paddle1YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(screen, WHITE, paddle1)

    def draw_paddle_2(paddle2YPos):
        """
        DOCSTRING
        """
        paddle2 = pygame.Rect(
            WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(screen, WHITE, paddle2)

    def update_ball(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection):
        """
        update the ball, using the paddle posistions the balls positions and the balls directions
        """
        ballXPos = ballXPos + ballXDirection * BALL_X_SPEED
        ballYPos = ballYPos + ballYDirection * BALL_Y_SPEED
        score = 0
        if (ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH and
            ballYPos + BALL_HEIGHT >= paddle1YPos and
            ballYPos - BALL_HEIGHT <= paddle1YPos + PADDLE_HEIGHT):
            ballXDirection = 1
        elif (ballXPos <= 0):
            ballXDirection = 1
            score = -1
            return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]
        if (ballXPos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and
            ballYPos + BALL_HEIGHT >= paddle2YPos and
            ballYPos - BALL_HEIGHT <= paddle2YPos + PADDLE_HEIGHT):
            ballXDirection = -1
        elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
            ballXDirection = -1
            score = 1
            return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]
        if (ballYPos <= 0):
            ballYPos = 0;
            ballYDirection = 1;
        elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
            ballYPos = WINDOW_HEIGHT - BALL_HEIGHT
            ballYDirection = -1
        return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

    def update_paddle_1(action, paddle1YPos):
        """
        DOCSTRING
        """
        if (action[1] == 1):
            paddle1YPos = paddle1YPos - PADDLE_SPEED
        if (action[2] == 1):
            paddle1YPos = paddle1YPos + PADDLE_SPEED
        if (paddle1YPos < 0):
            paddle1YPos = 0
        if (paddle1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
            paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
        return paddle1YPos

    def update_paddle_2(paddle2YPos, ballYPos):
        """
        DOCSTRING
        """
        if (paddle2YPos + PADDLE_HEIGHT/2 < ballYPos + BALL_HEIGHT/2):
            paddle2YPos = paddle2YPos + PADDLE_SPEED
        if (paddle2YPos + PADDLE_HEIGHT/2 > ballYPos + BALL_HEIGHT/2):
            paddle2YPos = paddle2YPos - PADDLE_SPEED
        if (paddle2YPos < 0):
            paddle2YPos = 0
        if (paddle2YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
            paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
        return paddle2YPos

class PongGame:
    """
    Game Class
    """
    def __init__(self):
        num = random.randint(0, 9)
        self.tally = 0
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.ballXDirection = 1
        self.ballYDirection = 1
        self.ballXPos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2
        if(0 < num < 3):
            self.ballXDirection = 1
            self.ballYDirection = 1
        if (3 <= num < 5):
            self.ballXDirection = -1
            self.ballYDirection = 1
        if (5 <= num < 8):
            self.ballXDirection = 1
            self.ballYDirection = -1
        if (8 <= num < 10):
            self.ballXDirection = -1
            self.ballYDirection = -1
        num = random.randint(0, 9)
        self.ballYPos = num*(WINDOW_HEIGHT - BALL_HEIGHT) / 9

    def get_next_frame(self, action):
        """
        update our screen
        """
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        self.paddle1YPos = update_paddle_1(action, self.paddle1YPos)
        draw_paddle_1(self.paddle1YPos)
        self.paddle2YPos = update_paddle_2(self.paddle2YPos, self.ballYPos)
        draw_paddle_2(self.paddle2YPos)
        [score,
         self.paddle1YPos,
         self.paddle2YPos,
         self.ballXPos,
         self.ballYPos,
         self.ballXDirection, self.ballYDirection] = update_ball(
             self.paddle1YPos,
             self.paddle2YPos,
             self.ballXPos,
             self.ballYPos,
             self.ballXDirection,
             self.ballYDirection)
        draw_ball(self.ballXPos, self.ballYPos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        self.tally = self.tally + score
        print("Tally is " + str(self.tally))
        return [score, image_data]

    def get_present_frame(self):
        """
        DOCSTRING
        """
        pygame.event.pump()
        screen.fill(BLACK)
        draw_paddle_1(self.paddle1YPos)
        draw_paddle_2(self.paddle2YPos)
        draw_ball(self.ballXPos, self.ballYPos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return image_data

class RL:
    """
    DOCSTRING
    """
    def __init__(self):
        self.actions = 3
        self.batch = 100
        self.explore = 500000 
        self.final_epsilon = 0.05
        self.gamma = 0.99
        self.initial_epsilon = 1.0
        self.observe = 50000
        self.replay_memory = 500000

    def create_graph(self):
        """
        Create TensorFlow graph.
        """
        W_conv1 = tensorflow.Variable(tensorflow.zeros([8, 8, 4, 32]))
        b_conv1 = tensorflow.Variable(tensorflow.zeros([32]))
        W_conv2 = tensorflow.Variable(tensorflow.zeros([4, 4, 32, 64]))
        b_conv2 = tensorflow.Variable(tensorflow.zeros([64]))
        W_conv3 = tensorflow.Variable(tensorflow.zeros([3, 3, 64, 64]))
        b_conv3 = tensorflow.Variable(tensorflow.zeros([64]))
        W_fc4 = tensorflow.Variable(tensorflow.zeros([3136, 784]))
        b_fc4 = tensorflow.Variable(tensorflow.zeros([784]))
        W_fc5 = tensorflow.Variable(tensorflow.zeros([784, self.actions]))
        b_fc5 = tensorflow.Variable(tensorflow.zeros([self.actions]))
        s = tensorflow.placeholder("float", [None, 84, 84, 4])
        conv1 = tensorflow.nn.relu(tensorflow.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1)
        conv2 = tensorflow.nn.relu(tensorflow.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2)
        conv3 = tensorflow.nn.relu(tensorflow.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
        conv3_flat = tensorflow.reshape(conv3, [-1, 3136])
        fc4 = tensorflow.nn.relu(tensorflow.matmul(conv3_flat, W_fc4) + b_fc4)
        fc5 = tensorflow.matmul(fc4, W_fc5) + b_fc5
        return s, fc5

    def train_graph(self, inp, out, sess):
        """
        Deep-Q network. Feed in pixel data to graph session.
        """
        argmax = tensorflow.placeholder('float', [None, self.actions]) 
        gt = tensorflow.placeholder('float', [None])
        action = tensorflow.reduce_sum(tensorflow.mul(out, argmax), reduction_indices=1)
        cost = tensorflow.reduce_mean(tensorflow.square(action - gt))
        train_step = tensorflow.train.AdamOptimizer(1e-6).minimize(cost)
        game = PongGame()
        D = collections.deque()
        frame = game.getPresentFrame()
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        inp_t = numpy.stack((frame, frame, frame, frame), axis=2)
        saver = tensorflow.train.Saver()
        sess.run(tensorflow.initialize_all_variables())
        t = 0
        epsilon = self.initial_epsilon
        while(1):
            out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
            argmax_t = numpy.zeros([self.actions])
            if(random.random() <= epsilon):
                maxIndex = random.randrange(self.actions)
            else:
                maxIndex = numpy.argmax(out_t)
            argmax_t[maxIndex] = 1
            if epsilon > self.final_epsilon:
                epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
            reward_t, frame = game.getNextFrame(argmax_t)
            frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
            ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
            frame = numpy.reshape(frame, (84, 84, 1))
            inp_t1 = numpy.append(frame, inp_t[:, :, 0:3], axis = 2)
            D.append((inp_t, argmax_t, reward_t, inp_t1))
            if len(D) > self.replay_memory:
                D.popleft()
            if t > self.observe:
                minibatch = random.sample(D, self.batch)
                inp_batch = [d[0] for d in minibatch]
                argmax_batch = [d[1] for d in minibatch]
                reward_batch = [d[2] for d in minibatch]
                inp_t1_batch = [d[3] for d in minibatch]
                gt_batch = list()
                out_batch = out.eval(feed_dict = {inp: inp_t1_batch})
                for i in range(len(minibatch)):
                    gt_batch.append(reward_batch[i] + self.gamma * numpy.max(out_batch[i]))
                train_step.run(feed_dict = {
                    gt: gt_batch, argmax: argmax_batch, inp: inp_batch})
            inp_t = inp_t1
            t += 1
            if t % 10000 == 0:
                saver.save(sess, './' + 'pong' + '-dqn', global_step = t)
            print("TIMESTEP / EPSILON / ACTION / REWARD / Q_MAX".format(
                t, epsilon, maxIndex, reward_t, numpy.max(out_t)))

if __name__ == '__main__':
    sess = tensorflow.InteractiveSession()
    inp, out = RL.create_graph()
    RL.train_graph(inp, out, sess)
