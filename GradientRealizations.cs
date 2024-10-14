using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class AbGradientDescent
    {
        protected NeuralNetwork currentNN;
        protected List<Layer> neuronLayer;
        protected Topology topology;
        protected IOptimizer Optimizer;
        public abstract double Backpropagation<T>(T dataset, int[] expected, int epoch);
        public AbGradientDescent(NeuralNetwork nn, IOptimizer optimizer = null)
        {
            if (optimizer == null)
            {
                this.Optimizer = new DefaultOptimizer();
            }
            else
            {
                this.Optimizer = optimizer;
            }
            currentNN = nn;
            neuronLayer = currentNN.neuronLayer;
            topology = currentNN.Topology;
        }
    }

    public class MiniBatchGD : AbGradientDescent
    {
        Random random;
        double LearningRate;
        int Capacity;
        public MiniBatchGD(NeuralNetwork nn, int capacity = 2, IOptimizer optimizer = null) : base(nn, optimizer) 
        {
            Capacity = capacity; 
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is List<double> outputsL)
            {
                double error = 0;
                for (int i = 0; i < error; i++)
                {
                    error += Backpropagation(1, outputsL, expected[0]);
                }
                return error;
            }
            else if (dataset is double[][] outputsM)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    for (int j = 0; j < Capacity; j++)
                    {
                        int rnd = random.Next(0, expected.Length - 1);
                        int ex = expected[rnd];
                        error += Backpropagation(i + 1, outputsM[rnd], ex);
                    }
                }
                return error * (outputsM.Length / Capacity) / epoch;
            }
            else
            {
                return 0;
            }
        }

        public double Backpropagation(int n, List<double> dataset, int expected)
        {
            double error = currentNN.FeedForward(dataset) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            LearningRate = topology.LearningRate / Math.Pow(n, 0.3);

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, LearningRate);
                }
            }
            return error;
        }

        public double Backpropagation(int n, double[] dataset, int expected)
        {
            double error = currentNN.FeedForward(dataset) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate);
                }
            }
            return error;
        }
    }

    public class StochasticGradient : AbGradientDescent
    {
        Random random;
        double LearningRate;
        public StochasticGradient(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is List<double> outputsL)
            {
                double error = 0;
                for (int i = 0; i < error; i++)
                {
                    error += Backpropagation(1, outputsL, expected[0]);
                }
                return error;
            }
            else if (dataset is double[][] outputsM)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    int rnd = random.Next(0, expected.Length - 1);
                    int ex = expected[rnd];
                    error += Backpropagation(i + 1, outputsM[rnd], ex);
                }
                return error * outputsM.Length / epoch;
            }
            else
            {
                return 0;
            }
        }

        public double Backpropagation(int n, List<double> dataset, int expected)
        {
            double error = currentNN.FeedForward(dataset) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            outputs = previousLayer.GetSignals();

            LearningRate = topology.LearningRate / Math.Pow(n, 0.3);

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, LearningRate);
                }
            }
            return error;
        }

        public double Backpropagation(int n, double[] dataset, int expected)
        {
            double error = currentNN.FeedForward(dataset) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate);
                }
            }
            return error;
        }
    }

    public class GradientDescent : AbGradientDescent
    {
        public GradientDescent(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is List<double> outputsL)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    error += Backpropagation(outputsL, expected[0]);
                }
                return error;
            }
            else if (dataset is double[][] outputsM)
            {
                double error = 0;
                for (int j = 0; j < epoch; j++)
                {
                    for (int i = 0; i < outputsM.Length; i++)
                    {
                        error += Backpropagation(outputsM[i], expected[i]);
                    }
                }
                return error / epoch;
            }
            else
            {
                return 0;
            }
        }

        public double Backpropagation(List<double> dataset, int expected)
        {
            double error = currentNN.FeedForward(dataset) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate);
                }
            }
            return error;
        }

        public double Backpropagation(double[] dataset, int expected)
        {
            double error = currentNN.FeedForward(dataset) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate);
                }
            }
            return error;
        }
    }
}
