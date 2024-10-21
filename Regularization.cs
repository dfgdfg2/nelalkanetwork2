using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public interface Regularization
    {
        double GetError(NeuralNetwork network, double[] dataset, int expected);
        double GetError(NeuralNetwork network, List<double> dataset, int expected);
    }

    public class DefaultReg : Regularization
    {
        public double GetError(NeuralNetwork network, double[] dataset, int expected)
        {
            return network.FeedForward(dataset) - expected;
        }
        public double GetError(NeuralNetwork network, List<double> dataset, int expected)
        {
            return network.FeedForward(dataset) - expected;
        }
    }

    public class L1 : Regularization
    {
        double lambda = 10 * Math.Pow(10, -1);
        public double GetError(NeuralNetwork network, double[] dataset, int expected)
        {
            return (network.FeedForward(dataset) - expected) + Math.Sqrt(Math.Pow(network.GetSumWeights(), 2)) * lambda;
        }
        public double GetError(NeuralNetwork network, List<double> dataset, int expected)
        {
            return (network.FeedForward(dataset) - expected) + Math.Sqrt(Math.Pow(network.GetSumWeights(), 2)) * lambda;
        }
    }

    public class L2 : Regularization
    {
        double lambda = 10 * Math.Pow(10, -3);
        public double GetError(NeuralNetwork network, double[] dataset, int expected)
        {
            return (network.FeedForward(dataset) - expected) + lambda * Math.Pow(network.GetSumWeights(), 2);
        }
        public double GetError(NeuralNetwork network, List<double> dataset, int expected)
        {
            return (network.FeedForward(dataset) - expected) + lambda * Math.Pow(network.GetSumWeights(), 2);
        }
    }
}

