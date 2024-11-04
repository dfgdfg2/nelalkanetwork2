using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public delegate double DelegateforWeights(double weight);

    public interface ITestRegularization
    {
        double FeedForward(List<double> inputs, List<Layer> neuronLayer);
        double FeedForward(double[] inputs, List<Layer> neuronLayer);
    }
    public interface ILearnRegularization
    {
        double FeedForward(List<double> inputs, List<Layer> neuronLayer);
        double FeedForward(double[] inputs, List<Layer> neuronLayer);
    }

    /* public class DefaultRegularization : RegularizationsForNN
    {
        DefaultOptimizer opti = new DefaultOptimizer();
        public DefaultRegularization(int fullEpoch) : base(fullEpoch) { }
        public DefaultRegularization()
        {

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
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }

        public override double FeedForward(List<double> inputs, List<Layer> neuronLayer)
        {
            { 
                Neuron currentNeuron = null;
                List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

                for (int i = 1; i < neuronLayer.Count; i++)
                {
                    double[] outputs = new double[neuronLayer[i].Count];
                    for (int j = 0; j < neuronLayer[i].Count; j++)
                    {
                        currentNeuron = neuronLayer[i][j];
                        double output = FeedForward(currentOutputs, currentNeuron);
                        outputs[j] = output;
                    }
                    currentOutputs = new List<double>(outputs);
                }
                return currentNeuron.Output;
            }
        }

        public override double FeedForward(double[] inputs, List<Layer> neuronLayer)
        {
            {
                Neuron currentNeuron = null;
                List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

                for (int i = 1; i < neuronLayer.Count; i++)
                {
                    double[] outputs = new double[neuronLayer[i].Count];
                    for (int j = 0; j < neuronLayer[i].Count; j++)
                    {
                        currentNeuron = neuronLayer[i][j];
                        double output = FeedForward(currentOutputs, currentNeuron);
                        outputs[j] = output;
                    }
                    currentOutputs = new List<double>(outputs);
                }
                return currentNeuron.Output;
            }
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, List<double> inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForward(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, double[] inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForward(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        public double FeedForward(double[] inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }
    }
    */

    public class DefaultRegularization : ITestRegularization, ILearnRegularization
    {
        IMetric opti = new MSE();

        public double FeedForward(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }

        public double FeedForward(List<double> inputs, List<Layer> neuronLayer)
        {
            {
                Neuron currentNeuron = null;
                List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

                for (int i = 1; i < neuronLayer.Count; i++)
                {
                    double[] outputs = new double[neuronLayer[i].Count];
                    for (int j = 0; j < neuronLayer[i].Count; j++)
                    {
                        currentNeuron = neuronLayer[i][j];
                        double output = FeedForward(currentOutputs, currentNeuron);
                        outputs[j] = output;
                    }
                    currentOutputs = new List<double>(outputs);
                }
                return currentNeuron.Output;
            }
        }

        public double FeedForward(double[] inputs, List<Layer> neuronLayer)
        {
            {
                Neuron currentNeuron = null;
                List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

                for (int i = 1; i < neuronLayer.Count; i++)
                {
                    double[] outputs = new double[neuronLayer[i].Count];
                    for (int j = 0; j < neuronLayer[i].Count; j++)
                    {
                        currentNeuron = neuronLayer[i][j];
                        double output = FeedForward(currentOutputs, currentNeuron);
                        outputs[j] = output;
                    }
                    currentOutputs = new List<double>(outputs);
                }
                return currentNeuron.Output;
            }
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, List<double> inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForward(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, double[] inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForward(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        public double FeedForward(double[] inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }
    }

    public class ConnectedChance
    {
        public const double usedChance = 0.4;
    }

    public class RegDropoutTest : ConnectedChance, ITestRegularization
    {
        IMetric opti = new MSE();
        public double FeedForwardTest(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i] * (1 - usedChance);
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }
        public double FeedForwardUsual(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }
        public double FeedForward(List<double> inputs, List<Layer> neuronLayer)
        {
            Neuron currentNeuron = null;
            List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

            int i = 0;
            double[] outputs = null;
            Layer currentLayer = null;
            for (i = 1; i < neuronLayer.Count - 1; i++)
            {
                outputs = new double[neuronLayer[i].Count];
                currentLayer = neuronLayer[i];
                for (int j = 0; j < outputs.Length; j++)
                {
                    currentNeuron = currentLayer[j];
                    double output = FeedForwardTest(currentOutputs, currentNeuron);
                    outputs[j] = output;
                }
                currentOutputs = new List<double>(outputs);
            }
            outputs = new double[neuronLayer[i].Count];
            currentLayer = neuronLayer[i];
            for (int j = 0; j < outputs.Length; j++)
            {
                currentNeuron = currentLayer[j];
                double output = FeedForwardUsual(currentOutputs, currentNeuron);
                outputs[j] = output;
            }
            return currentNeuron.Output;
        }

        public double FeedForward(double[] inputs, List<Layer> neuronLayer)
        {
            Neuron currentNeuron = null;
            List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

            int i = 0;
            double[] outputs = null;
            Layer currentLayer = null;
            for (i = 1; i < neuronLayer.Count - 1; i++)
            {
                outputs = new double[neuronLayer[i].Count];
                currentLayer = neuronLayer[i];
                for (int j = 0; j < outputs.Length; j++)
                {
                    currentNeuron = currentLayer[j];
                    double output = FeedForwardTest(currentOutputs, currentNeuron);
                    outputs[j] = output;
                }
                currentOutputs = new List<double>(outputs);
            }
            outputs = new double[neuronLayer[i].Count];
            currentLayer = neuronLayer[i];
            for (int j = 0; j < outputs.Length; j++)
            {
                currentNeuron = currentLayer[j];
                double output = FeedForwardUsual(currentOutputs, currentNeuron);
                outputs[j] = output;
            }
            return currentNeuron.Output;
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, double[] inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForwardUsual(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, List<double> inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForwardUsual(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }
    }

    public class RegDropoutLearn : ConnectedChance, ILearnRegularization
    {
        private bool Dropout()
        {
            if (random.NextDouble() > usedChance)
            {
                return true;
            }
            return false;
        }
        IMetric opti = new MSE();
        Random random = new Random();
        public double FeedForwardLearn(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            if (Dropout())
            {
                for (int i = 0; i < neuron.WeightsCount; i++)
                {
                    output += neuron.Weights[i] * inputs[i];
                }
                if (neuron.neuronType != NeuronType.Input)
                {
                    neuron.Output = opti.Sigmoid(output);
                    return neuron.Output;
                }
                neuron.Output = output;
                return output;
            }
            neuron.Output = output;
            return output;
        }
        public double FeedForwardUsual(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            if (neuron.neuronType != NeuronType.Input)
            {
                neuron.Output = opti.Sigmoid(output);
                return neuron.Output;
            }
            neuron.Output = output;
            return output;
        }
        public double FeedForward(List<double> inputs, List<Layer> neuronLayer)
        {
            Neuron currentNeuron = null;
            List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

            int i = 0;
            double[] outputs = null;
            Layer currentLayer = null;
            for (i = 1; i < neuronLayer.Count - 1; i++)
            {
                outputs = new double[neuronLayer[i].Count];
                currentLayer = neuronLayer[i];
                for (int j = 0; j < outputs.Length; j++)
                {
                    currentNeuron = currentLayer[j];
                    double output = FeedForwardLearn(currentOutputs, currentNeuron);
                    outputs[j] = output;
                }
                currentOutputs = new List<double>(outputs);
            }
            outputs = new double[neuronLayer[i].Count];
            currentLayer = neuronLayer[i];
            for (int j = 0; j < outputs.Length; j++)
            {
                currentNeuron = currentLayer[j];
                double output = FeedForwardUsual(currentOutputs, currentNeuron);
                outputs[j] = output;
            }
            return currentNeuron.Output;
        }
        public double FeedForward(double[] inputs, List<Layer> neuronLayer)
        {
            Neuron currentNeuron = null;
            List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);

            int i = 0;
            double[] outputs = null;
            Layer currentLayer = null;
            for (i = 1; i < neuronLayer.Count - 1; i++)
            {
                outputs = new double[neuronLayer[i].Count];
                currentLayer = neuronLayer[i];
                for (int j = 0; j < outputs.Length; j++)
                {
                    currentNeuron = currentLayer[j];
                    double output = FeedForwardLearn(currentOutputs, currentNeuron);
                    outputs[j] = output;
                }
                currentOutputs = new List<double>(outputs);
            }
            outputs = new double[neuronLayer[i].Count];
            currentLayer = neuronLayer[i];
            for (int j = 0; j < outputs.Length; j++)
            {
                currentNeuron = currentLayer[j];
                double output = FeedForwardUsual(currentOutputs, currentNeuron);
                outputs[j] = output;
            }
            return currentNeuron.Output;
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, List<double> inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForwardUsual(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        private List<double> FeedForwardInputs(Neuron currentNeuron, double[] inputs, List<Layer> neuronLayer)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = FeedForwardUsual(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }
    }



    public class Regularization
    {
        Random random = new Random();
        static double LambdaL1 = Math.Pow(10, -6);
        static double LambdaL2 = Math.Pow(10, -4);
        public double L1(double weight)
        {
            return LambdaL1 * Math.Abs(weight);
        }

        public double L2(double weight)
        {
            return LambdaL2 * weight * 2;
        }

        public double L1andL2(double weight)
        {
            return L1(weight) + L2(weight);
        }
    }

}

