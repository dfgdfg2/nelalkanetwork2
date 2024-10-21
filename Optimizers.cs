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
        double SigmoidDX(double value);
        double Sigmoid(double value);
        double GetFromAllDelta(Layer forwardLayer, int j);
        
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
            double delta = error * SigmoidDX(neuron.Output);
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
            double delta = error * SigmoidDX(neuron.Output);
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double output = previousOutputs[i];
                neuron.Weights[i] = currentWeight - learningRate * delta * output;
            }
            return delta;
        }

        public double GetFromAllDelta(Layer forwardLayer, int j)
        {
            double deltaSum = 0;
            for (int k = 0; k < forwardLayer.Count; k++)
            {
                Neuron neuron = forwardLayer.Neurons[k];
                double delta = neuron.Delta;
                double weights = neuron.Weights[j];
                deltaSum += delta * weights;
            }
            return deltaSum;
        }

        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal * (1 - sigmVal);
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
        protected double[][][] Inertions { get; set; }
        protected double Koef { get; set; }
        public Momentum(Topology topology)
        {
            Inertions = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                Inertions[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    Inertions[i][j] = new double[weights];
                }
            }
            Koef = 0.9;
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
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i] * Koef + delta * learningRate * previousOutputs[i]; // исправить error на delta в случае нестабильности!!!!!!!!!!!
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = currentWeight - inertia;
            }
            return delta;
        }

        public double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate)
        {
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i] * Koef + delta * learningRate * previousOutputs[i];
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = currentWeight - inertia;
            }
            return delta;
        }

        public double GetFromAllDelta(Layer forwardLayer, int j)
        {
            double deltaSum = 0;
            for (int k = 0; k < forwardLayer.Count; k++)
            {
                Neuron neuron = forwardLayer.Neurons[k];
                double delta = neuron.Delta;
                double weights = neuron.Weights[j];
                deltaSum += delta * weights;
            }
            return deltaSum;
        }

        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal * (1 - sigmVal);
        }

        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
    }
    //NOT DEFINED
    public class NesterovMomentum : Momentum, IOptimizer
    {
        public NesterovMomentum(Topology topology) : base(topology)
        {

        }
        public new double BackPropagation(Neuron neuron, List<double> previousOutputs, double error, double learningRate)
        {
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;

            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double currentWeight = neuron.Weights[i];
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i];
                double predictedWeight = currentWeight - inertia * Koef;
                //inertia = inertia * Koef + delta * learningRate * previousOutputs[i];
                inertia = inertia * Koef + delta * learningRate * (previousOutputs[i] - Koef * inertia);
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = predictedWeight - inertia;
            }

            return delta;
        }
        public new double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate)
        {
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {

                double currentWeight = neuron.Weights[i];
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i];
                double predictedWeight = currentWeight - inertia * Koef;
                //inertia = inertia * Koef + delta * learningRate * previousOutputs[i];
                inertia = inertia * Koef + delta * learningRate * (previousOutputs[i] - Koef * inertia);
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = predictedWeight - inertia;

                /*
                double currentWeight = neuron.Weights[i];
        
                // Обновляем инерцию с учетом предыдущих значений
                double inertia = Inertions[numberLayer][neuron.NumberOfLayer][i];
                double predictedWeight = currentWeight - Koef * inertia; // Предсказанные веса

                // Теперь вычисляем новую инерцию
                inertia = inertia * Koef + delta * learningRate * previousOutputs[i];
                Inertions[numberLayer][neuron.NumberOfLayer][i] = inertia;

                // Обновляем вес нейрона
                neuron.Weights[i] = predictedWeight - inertia;
                */
            }
            return delta;
        }

        public new double GetFromAllDelta(Layer forwardLayer, int j)
        {
            double deltaSum = 0;
            for (int k = 0; k < forwardLayer.Count; k++)
            {
                Neuron neuron = forwardLayer.Neurons[k];
                double delta = neuron.Delta;
                double weights = neuron.Weights[j];// - (Inertions[neuron.NumberLayer - 1][neuron.NumberOfLayer][j] * Koef); 
                deltaSum += delta * weights;
            }
            return deltaSum;
        }
    }

    public class RMSProp : IOptimizer
    {
        protected double[][][] G { get; set; }
        double E = Math.Pow(10, -9);
        double Alpha = 0.9;
        public RMSProp(Topology topology)
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
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
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = previousOutputs[i];
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                g = Alpha * g + (1 - Alpha) * Math.Pow(delta * output, 2);
                double currentWeight = neuron.Weights[i];
                neuron.Weights[i] = currentWeight - delta * output * learningRate / Math.Sqrt(g + E);
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
            }
            return delta;
        }

        public double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate)
        {
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = previousOutputs[i];
                double g = G[numberLayer][neuron.NumberOfLayer][i];
                g = Alpha* g + (1 - Alpha) * Math.Pow(delta * output, 2);
                double currentWeight = neuron.Weights[i]; 
                neuron.Weights[i] = currentWeight - (delta * previousOutputs[i] * learningRate) / Math.Sqrt(g + E);
                G[numberLayer][neuron.NumberOfLayer][i] = g;
            }
            return delta;
        }

        public double GetFromAllDelta(Layer forwardLayer, int j)
        {
            double deltaSum = 0;
            for (int k = 0; k < forwardLayer.Count; k++)
            {
                Neuron neuron = forwardLayer.Neurons[k];
                double delta = neuron.Delta;
                double weights = neuron.Weights[j];
                deltaSum += delta * weights;
            }
            return deltaSum;
        }

        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal * (1 - sigmVal);
        }

        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
    }

    public class Adagrad : IOptimizer
    {
        protected double[][][] G { get; set; }
        double E = Math.Pow(10, -9);
        public Adagrad(Topology topology)
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
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
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = previousOutputs[i];
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                double weightDelta = delta * output;
                g = g + Math.Pow(weightDelta, 2);
                double currentWeight = neuron.Weights[i];
                neuron.Weights[i] = currentWeight - weightDelta * learningRate / Math.Sqrt(g + E);
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
            }
            return delta;
        }

        public double BackPropagation(Neuron neuron, double[] previousOutputs, double error, double learningRate)
        {
            double delta = error * SigmoidDX(neuron.Output);
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = previousOutputs[i];
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                double weightDelta = delta * output;
                g = g + Math.Pow(weightDelta, 2);
                double currentWeight = neuron.Weights[i];
                neuron.Weights[i] = currentWeight - weightDelta * learningRate / Math.Sqrt(g + E);
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
            }
            return delta;
        }

        public double GetFromAllDelta(Layer forwardLayer, int j)
        {
            double deltaSum = 0;
            for (int k = 0; k < forwardLayer.Count; k++)
            {
                Neuron neuron = forwardLayer.Neurons[k];
                double delta = neuron.Delta;
                double weights = neuron.Weights[j];
                deltaSum += delta * weights;
            }
            return deltaSum;
        }

        public double SigmoidDX(double value)
        {
            double sigmVal = Sigmoid(value);
            return sigmVal * (1 - sigmVal);
        }

        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }
    }
}
