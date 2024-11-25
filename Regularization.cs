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

    public interface IAggRegularization 
    {
        ILearnRegularization LearnRegularization { set; get; }
        ITestRegularization TestRegularization { set; get; }
    }

    public class AggDefaultRegularization : IAggRegularization
    {
        public ILearnRegularization LearnRegularization { set; get; }
        public ITestRegularization TestRegularization { set; get; }

        public AggDefaultRegularization()
        {
            LearnRegularization = new DefaultRegularization();
            TestRegularization = new DefaultRegularization();
        }
    }

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

    public class AggDropout : IAggRegularization
    {
        public ILearnRegularization LearnRegularization { set; get; }
        public ITestRegularization TestRegularization { set; get; }

        public AggDropout() 
        {
            LearnRegularization = new RegDropoutLearn();
            TestRegularization = new RegDropoutTest();
        }
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

    public class BatchParams
    {
        public List<double> Gamma;
        public double[] Beta;
        public double E = Math.Pow(10, -6);
        public double Alpha = 0.9;

        public BatchParams(int layers) 
        {
            Gamma = new List<double>();
            for (int i = 0; i < layers; i++)
            {
                Gamma.Add(1);
            }
            Beta = new double[layers];
        }
    }

    public class AggBatchNormalization : IAggRegularization
    {
        public ILearnRegularization LearnRegularization { set; get; }
        public ITestRegularization TestRegularization { set; get; }

        public AggBatchNormalization(BatchParams bp)
        {
            LearnRegularization = new BatchNormalizationLearn(bp);
            TestRegularization = new BatchNormalizationLearn(bp);
        }
    }

    public class BatchNormalizationLearn : ILearnRegularization, ITestRegularization
    {
        public BatchParams bp;
        IMetric opti = new MSE();
        public BatchNormalizationLearn(BatchParams bp)
        {
            this.bp = bp;
        }

        public double FeedForward(List<double> inputs, Neuron neuron)
        {
            double output = 0;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                output += neuron.Weights[i] * inputs[i];
            }
            return output;
        }

        private void NormalizeOutputs(List<double> outputs, List<Neuron> currentLayer, int j)
        {
            double currentGamma = bp.Gamma[j];
            double currentBeta = bp.Beta[j];
            double average = 0;
            double dispersion = 0;
            for (int i = 0; i < outputs.Count; i++)
            {
                average += outputs[i];
            }
            average = average / outputs.Count;
            for (int i = 0; i < outputs.Count; i++)
            {
                dispersion += Math.Pow(outputs[i] - average, 2);
            }
            dispersion = dispersion / outputs.Count;
            for (int i = 0; i < outputs.Count; i++)
            {
                currentLayer[i].Output = opti.Sigmoid(currentGamma * ((outputs[i] - average) / Math.Sqrt(dispersion + bp.E)) + currentBeta);
            }
        }

        private void NormalizeInputOutputs(List<double> outputs, List<Neuron> currentLayer, int j)
        {
            double currentGamma = bp.Gamma[j];
            double currentBeta = bp.Beta[j];
            double average = 0;
            double dispersion = 0;
            for (int i = 0; i < outputs.Count; i++)
            {
                average += outputs[i];
            }
            average = average / outputs.Count;
            for (int i = 0; i < outputs.Count; i++)
            {
                dispersion += Math.Pow(outputs[i] - average, 2);
            }
            dispersion = dispersion / outputs.Count;
            for (int i = 0; i < outputs.Count; i++)
            {
                currentLayer[i].Output = currentGamma * ((outputs[i] - average) / Math.Sqrt(dispersion + bp.E)) + currentBeta;
            }
        }

        public double FeedForward(List<double> inputs, List<Layer> neuronLayer)
        {
            {
                Neuron currentNeuron = null;
                List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);
                NormalizeInputOutputs(currentOutputs, neuronLayer[0].Neurons, 0);

                for (int i = 1; i < neuronLayer.Count; i++)
                {
                    double[] outputs = new double[neuronLayer[i].Count];
                    Layer currentLayer = neuronLayer[i];
                    for (int j = 0; j < currentLayer.Count; j++)
                    {
                        currentNeuron = currentLayer[j];
                        double output = FeedForward(currentOutputs, currentNeuron);
                        outputs[j] = output;
                    }
                    currentOutputs = new List<double>(outputs);
                    NormalizeOutputs(currentOutputs, neuronLayer[i].Neurons, i);
                }
                return currentNeuron.Output;
            }
        }

        public double FeedForward(double[] inputs, List<Layer> neuronLayer)
        {
            {
                Neuron currentNeuron = null;
                List<double> currentOutputs = FeedForwardInputs(currentNeuron, inputs, neuronLayer);
                NormalizeInputOutputs(currentOutputs, neuronLayer[0].Neurons, 0);

                for (int i = 1; i < neuronLayer.Count; i++)
                {
                    double[] outputs = new double[neuronLayer[i].Count];
                    Layer currentLayer = neuronLayer[i];
                    for (int j = 0; j < currentLayer.Count; j++)
                    {
                        currentNeuron = currentLayer[j];
                        double output = FeedForward(currentOutputs, currentNeuron);
                        outputs[j] = output;
                    }
                    currentOutputs = new List<double>(outputs);
                    NormalizeOutputs(currentOutputs, neuronLayer[i].Neurons, i);
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
            return output;
        }
    }


    public abstract class AbRegularizationWeights
    {
        protected static double LambdaL1 = Math.Pow(10, -4);
        protected static double LambdaL2 = Math.Pow(10, -4);

        public abstract double Regulate(double weight);
    }

    public class L1 : AbRegularizationWeights
    {
        public double RegL1(double weight)
        {
            return LambdaL1 * Math.Sign(weight);
        }

        public override double Regulate(double weight)
        {
            return RegL1(weight);
        }
    }

    public class L2 : AbRegularizationWeights
    {
        public double RegL2(double weight)
        {
            return LambdaL2 * weight * 2;
        }
        public override double Regulate(double weight)
        {
            return RegL2(weight);
        }
    }

    public class ElasticNet : AbRegularizationWeights
    {
        public override double Regulate(double weight)
        {
            return LambdaL2 * weight * 2 + LambdaL1 * Math.Sign(weight);
        }
    }

    public class L0 : AbRegularizationWeights
    {
        public override double Regulate(double weight)
        {
            return 0;
        }
    }
}

