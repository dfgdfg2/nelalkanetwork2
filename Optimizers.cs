using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public interface IOptimizer
    {
        double FeedForward(List<double> inputs, Neuron neuron);
        double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate);
        double BackPropagation(Neuron neuron, List<double> previousOutputs, double error, double learningRate);
    }
    /// <summary>
    /// ////////////////////////////
    /// </summary>
    public class DefaultOptimizer : IOptimizer
    {
        public double FeedForward(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }

        public double BackPropagation(Neuron neuron, List<double> previousOutputs, double error, double learningRate)
        {
            double delta = 0;
            if (neuron.neuronType == NeuronType.Output)
            {
                delta = error * SigmoidDX(neuron.Output);
            }
            else
            {
                delta = error;
            }
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double output = previousOutputs[i];
                neuron.Weights[i] = currentWeight - learningRate * delta * output;
            }
            return delta;
        }

        public double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate)
        {
            double delta = 0;
            if (neuron.neuronType == NeuronType.Output)
            {
                delta = error * SigmoidDX(neuron.Output);
            }
            else
            {
                delta = error;
            }
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double output = previousOutputs[i];
                neuron.Weights[i] = currentWeight - learningRate * delta * output;
            }
            return delta;
        }

        public double GetDelta(double error, double output)
        {
            return error * Sigmoid(output);
        }

        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal / (1 - sigmVal);
        } 

        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
    }
    /// <summary>
    /// ////////////////////////////
    /// </summary>
    public class Momentum : IOptimizer
    {
        double[][][] Inertions { get; set; }
        public Momentum(Topology topology)
        {
            Inertions = new double[topology.Count][][];
            Inertions[0] = new double[topology.CollectionCounts[0]][];
            int count = topology.CollectionCounts[0];
            for (int i = 0; i < count; i++)
            {
                Inertions[0][i] = new double[1];
            }
            for (int i = 1; i < topology.CollectionCounts.Count; i++)
            {
                int collection = topology.CollectionCounts[i];
                Inertions[i] = new double[collection][];
                int weights = topology.CollectionCounts[i - 1];
                for (int j = 0; j < collection; j++)
                {
                    Inertions[i][j] = new double[weights];
                }
            }
        }
        public double FeedForward(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }

        public double BackPropagation(Neuron neuron, List<double> previousOutputs, double error, double learningRate)
        {
            double delta = 0;
            if (neuron.neuronType == NeuronType.Output)
            {
                delta = error * SigmoidDX(neuron.Output);
            }
            else
            {
                delta = error;
            }
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double inertia = Inertions[numberLayer][neuron.NumberOfLayer][i] * 0.9 + delta * learningRate * previousOutputs[i];
                Inertions[numberLayer][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = currentWeight - inertia;
            }
            return delta;
        }

        public double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate)
        {
            return 0;
        }

        public double GetDelta(double error, double output)
        {
            return error * Sigmoid(output);
        }

        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal / (1 - sigmVal);
        }

        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
    }
}
