using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Reflection;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ConsoleApp1
{
    public abstract class AbGradientDescent
    {
        protected NeuralNetwork currentNN;
        public ILearnRegularization regularization { get; set; }
        protected List<Layer> neuronLayer;
        protected Topology topology;
        protected IOptimizer Optimizer;
        protected int CurrentEpoch;
        protected IMetric metric = new MAE();
        public AbGradientDescent(NeuralNetwork nn, IOptimizer optimizer = null)
        {
            currentNN = nn;
            neuronLayer = currentNN.neuronLayer;
            topology = currentNN.Topology;
            if (optimizer == null)
            {
                optimizer = new DefaultOptimizer();
                Optimizer = optimizer;
            }
            else
            {
                Optimizer = optimizer;
            }
        }
        public abstract double Backpropagation<T>(T dataset, int[] expected, int epoch);
        public virtual void Backpropagation(/*double error, */double learningRate)
        {
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                //double delta = error * metric.SigmoidDX(currentNeuron.Output);
                Optimizer.BackPropagation(currentNeuron, outputs, learningRate, CurrentEpoch/*, delta*/);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                previousLayer = neuronLayer[i - 1];
                Layer forwardLayer = neuronLayer[i + 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    //double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    Neuron currentNeuron = currentLayer[j];
                    //deltaSum = deltaSum * metric.SigmoidDX(currentNeuron.Output);
                    Optimizer.BackPropagation(currentNeuron, outputs, learningRate, CurrentEpoch/*, deltaSum*/);
                }
            }
        }

        public virtual void BackpropagationWithBatchNorm(/*double error, */double learningRate)
        {
            BatchParams bp = currentNN.BatchParams;
            int count = neuronLayer.Count;
            double deltaSum = 0;
            double deltaOutput = 0;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                deltaSum += currentNeuron.Delta;
                deltaOutput += currentNeuron.Delta * currentNeuron.Output;
                //double delta = error * metric.SigmoidDX(currentNeuron.Output);
                Optimizer.BackPropagation(currentNeuron, outputs, learningRate, CurrentEpoch/*, delta*/);
            }

            bp.Beta[0] = bp.Beta[0] - deltaSum * learningRate;
            bp.Gamma[0] = bp.Gamma[0] - deltaOutput * learningRate;

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                previousLayer = neuronLayer[i - 1];
                //Layer forwardLayer = neuronLayer[i + 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    //double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    Neuron currentNeuron = currentLayer[j];
                    //deltaSum = deltaSum * metric.SigmoidDX(currentNeuron.Output);
                    Optimizer.BackPropagation(currentNeuron, outputs, learningRate, CurrentEpoch/*, deltaSum*/);
                    deltaSum += currentNeuron.Delta;
                    deltaOutput += currentNeuron.Delta * currentNeuron.Output;
                }
                bp.Beta[i] = bp.Beta[i] - deltaSum * learningRate;
                bp.Gamma[i] = bp.Gamma[i] - deltaOutput * learningRate;
            }
        }

        public virtual List<double> Deltapropagation(double error)
        {
            List<double> delta = new List<double>();
            Layer currentLayer = neuronLayer.Last();
            delta.AddRange(metric.SetFirstDelta(error, currentLayer));

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                Layer forwardLayer = neuronLayer[i + 1];
                currentLayer = neuronLayer[i];
                delta.AddRange(metric.SetDeltas(forwardLayer, currentLayer));
            }
            return delta;
        }

        protected double GetActual(double[] outputs)
        {
            double actual = regularization.FeedForward(outputs, currentNN.neuronLayer);
            return actual;
        }

        protected List<double> GetAverageDeltaWeights(List<List<double>> deltas)
        {
            List<double> averageDeltas = new List<double>();
            int countRow = deltas[0].Count;
            for (int i = 0; i < countRow; i++)
            {
                averageDeltas.Add(0);
                for (int j = 0; j < deltas.Count; j++)
                {
                    averageDeltas[i] += deltas[j][i];
                }
                averageDeltas[i] /= deltas.Count;
            }
            return averageDeltas;
        }

        protected List<double> CollectWeights(List<List<double>> deltas)
        {
            List<double> weights = new List<double>();
            int counts = currentNN.neuronLayer.Count;
            for (int k = 0; k < deltas.Count; k++)
            {
                List<double> delts = deltas[k];
                for (int g = 0; g < deltas[k].Count; g++)
                {
                    Layer currentL = currentNN.neuronLayer[counts - k - 2];
                    weights.AddRange(Optimizer.GetDeltaWeights(topology.LearningRate, delts[g], currentL.Neurons, g));
                }
            }
            return weights;
        }
    }

    public class GradientDescent : AbGradientDescent
    {
        public GradientDescent(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
        }
        protected List<double> WeightsGradient;
        public override void Backpropagation(double learningRate)
        {
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            int k = 0;
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                k = Optimizer.SetWeights(currentNeuron, WeightsGradient);
                WeightsGradient.RemoveRange(0, k);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    k = Optimizer.SetWeights(currentNeuron, WeightsGradient);
                    WeightsGradient.RemoveRange(0, k);
                }
            }
        }
        public new List<List<double>> Deltapropagation(double error)
        {
            List<List<double>> delta = new List<List<double>>();
            Layer currentLayer = neuronLayer.Last();
            delta.Add(metric.SetFirstDelta(error, currentLayer));

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                Layer forwardLayer = neuronLayer[i + 1];
                currentLayer = neuronLayer[i];
                delta.Add(metric.SetDeltas(forwardLayer, currentLayer));
            }
            return delta;
        }

        public List<double> DeltaWeightsPropagation(double error, double learningRate)
        {
            List<double> weights = new List<double>();
            Layer currentLayer = neuronLayer.Last();
            Layer forwardLayer = neuronLayer[neuronLayer.Count - 2];
            List<Neuron> neurons = forwardLayer.Neurons;
            weights.AddRange(metric.SetFirstDeltaWeights(error, learningRate, currentLayer, Optimizer, neurons));

            for(int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                forwardLayer = neuronLayer[i + 1];
                Layer previousLayer = neuronLayer[i - 1];
                weights.AddRange(metric.SetDeltaWeights(error, learningRate, currentLayer, Optimizer, forwardLayer, previousLayer.Neurons));
            }
            return weights;
        } 

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                if (regularization is BatchNormalizationLearn bn)
                {
                    double metr = 0;
                    for (int i = 0; i < epoch; i++)
                    {
                        CurrentEpoch++;
                        double metrtemp = 0;
                        List<List<double>> weights = new List<List<double>>();
                        for (int j = 0; j < expected.Length; j++)
                        {
                            int exp = expected[j];
                            double[] outp = outputsM[j];
                            double actual = GetActual(outp);
                            metrtemp += metric.GetMetric(actual, exp);
                            double error = actual - exp;
                            weights.Add(DeltaWeightsPropagation(error, topology.LearningRate));
                        }
                        metr += metrtemp / expected.Length;
                        WeightsGradient = GetAverageDeltaWeights(weights);
                        Backpropagation(currentNN.Topology.LearningRate);
                        Optimizer.Epoch += 1;
                    }
                    return metr / epoch * 100;
                }
                else
                {
                    double metr = 0;
                    for (int i = 0; i < epoch; i++)
                    {
                        CurrentEpoch++;
                        double metrtemp = 0;
                        List<List<double>> weights = new List<List<double>>();
                        for (int j = 0; j < expected.Length; j++)
                        {
                            int exp = expected[j];
                            double[] outp = outputsM[j];
                            double actual = GetActual(outp);
                            metrtemp += metric.GetMetric(actual, exp);
                            double error = actual - exp;
                            weights.Add(DeltaWeightsPropagation(error, topology.LearningRate));
                        }
                        metr += metrtemp / expected.Length;
                        WeightsGradient = GetAverageDeltaWeights(weights);
                        Backpropagation(currentNN.Topology.LearningRate);
                        Optimizer.Epoch += 1;
                    }
                    return metr / epoch * 100;
                }
            }
            else
            {
                return 0;
            }
        }

    }

    public class MiniBatchGD : GradientDescent
    {
        int Capacity;
        public MiniBatchGD(NeuralNetwork nn, int capacity = 2, IOptimizer optimizer = null) : base(nn, optimizer)
        {
            if (capacity < 1)
            {
                return;
            }
            Capacity = capacity;
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                if (regularization is BatchNormalizationLearn bn)
                {
                    double metr = 0;
                    for (int i = 0; i < epoch; i++)
                    {
                        CurrentEpoch++;
                        double metrtemp = 0;
                        UniqueRandom uniqueRandom = new UniqueRandom(0, expected.Length - 1);
                        for (int j = 0; j < expected.Length / Capacity; j++)
                        {
                            List<List<double>> weights = new List<List<double>>();
                            for (int k = 0; k < Capacity; k++)
                            {
                                int rnd = uniqueRandom.GetUniqueRandomNumber();
                                int exp = expected[rnd];
                                double[] outp = outputsM[rnd];
                                double actual = GetActual(outp);
                                metrtemp += metric.GetMetric(actual, exp);
                                double error = actual - exp;
                                weights.Add(DeltaWeightsPropagation(error, topology.LearningRate));
                            }
                            WeightsGradient = GetAverageDeltaWeights(weights);
                            Backpropagation(currentNN.Topology.LearningRate);
                            Optimizer.Epoch += 1;
                        }
                        metr += metrtemp / expected.Length;
                    }
                    return metr / epoch * 100;
                }
                else
                {
                    double metr = 0;
                    for (int i = 0; i < epoch; i++)
                    {
                        CurrentEpoch++;
                        double metrtemp = 0;
                        UniqueRandom uniqueRandom = new UniqueRandom(0, expected.Length - 1);
                        for (int j = 0; j < expected.Length / Capacity; j++)
                        {
                            List<List<double>> weights = new List<List<double>>();
                            for (int k = 0; k < Capacity; k++)
                            {
                                int rnd = uniqueRandom.GetUniqueRandomNumber();
                                int exp = expected[rnd];
                                double[] outp = outputsM[rnd];
                                double actual = GetActual(outp);
                                metrtemp += metric.GetMetric(actual, exp);
                                double error = actual - exp;
                                weights.Add(DeltaWeightsPropagation(error, topology.LearningRate));
                            }
                            WeightsGradient = GetAverageDeltaWeights(weights);
                            Backpropagation(currentNN.Topology.LearningRate);
                            Optimizer.Epoch += 1;
                        }
                        metr += metrtemp / expected.Length;
                    }
                    return metr / epoch * 100;
                }
            }
            else
            {
                return 0;
            }
        }
    }
    public class StochasticGradient : AbGradientDescent
    {
        public StochasticGradient(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                double metr = 0;
                for (int i = 0; i < epoch; i++)
                {
                    double metrtemp = 0;
                    UniqueRandom uniqueRandom = new UniqueRandom(0, expected.Length - 1);
                    CurrentEpoch++;
                    for (int j = 0; j < expected.Length; j++)
                    {
                        int numb = uniqueRandom.GetUniqueRandomNumber();
                        int exp = expected[numb];
                        double actual = GetActual(outputsM[numb]);
                        double error = actual - exp;
                        metrtemp += metric.GetMetric(actual, exp);
                        Deltapropagation(error);
                        Backpropagation(/*error, */topology.LearningRate);
                    }
                    metr += metrtemp / expected.Length;
                }
                return metr / epoch * 100;
            }
            else
            {
                return 0;
            }
        }
    }
}
