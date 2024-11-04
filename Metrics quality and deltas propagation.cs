using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class IMetric
    {
        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal * (1 - sigmVal);
        }
        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        public abstract List<double> SetFirstDelta(double error, Layer layer);
        public abstract List<double> SetDeltas(Layer forwardLayer, Layer layer);
        public abstract double GetMetric(double actual, double expected);
    }
    public class MSE : IMetric
    {
        public override List<double> SetFirstDelta(double error, Layer layer)
        {
            List<double> deltas = new List<double>();
            for (int i = 0; i < layer.Count; i++)
            {
                Neuron neuron = layer[i];
                neuron.Delta = 2 * error * SigmoidDX(neuron.Output);
            }
            return deltas;
        }

        public override List<double> SetDeltas(Layer forwardLayer, Layer layer)
        {
            List<double> deltas = new List<double>();
            for (int j = 0; j < layer.Count; j++)
            {
                Neuron neuron = layer[j];
                for (int k = 0; k < forwardLayer.Count; k++)
                {
                    Neuron forwardNeuron = forwardLayer[k];
                    double forwardWeight = forwardNeuron.Weights[j];
                    neuron.Delta += forwardNeuron.Delta * forwardWeight * 2;
                    deltas.Add(forwardNeuron.Delta);
                }
            }
            return deltas;
        }

        public override double GetMetric(double actual, double expected)
        {
            return Math.Pow(actual - expected, 2);
        }
    }

    public class MAE : IMetric
    {
        public override List<double> SetFirstDelta(double error, Layer layer)
        {
            List<double> deltas = new List<double>();
            for (int i = 0; i < layer.Count; i++)
            {
                Neuron neuron = layer[i];
                neuron.Delta = error * SigmoidDX(neuron.Output);
                deltas.Add(neuron.Delta);
            }
            return deltas;
        }

        public override List<double> SetDeltas(Layer forwardLayer, Layer layer)
        {
            List<double> deltas = new List<double>();
            for (int j = 0; j < layer.Count; j++)
            {
                Neuron neuron = layer[j];
                for (int k = 0; k < forwardLayer.Count; k++)
                {
                    Neuron forwardNeuron = forwardLayer[k];
                    double forwardWeight = forwardNeuron.Weights[j];
                    neuron.Delta += forwardNeuron.Delta * forwardWeight;
                    deltas.Add(neuron.Delta);
                }
                neuron.Delta = neuron.Delta * SigmoidDX(neuron.Output);
            }
            return deltas;
        }
        public override double GetMetric(double actual, double expected)
        {
            return Math.Abs(actual - expected);
        }
    }
}
