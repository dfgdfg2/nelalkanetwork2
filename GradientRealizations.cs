using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class AbGradientDescent
    {
        protected NeuralNetwork currentNN;
        public ILearnRegularization regularization { get; private set; }
        protected List<Layer> neuronLayer;
        protected Topology topology;
        protected IOptimizer Optimizer;
        protected List<Regularizations> regularizations;
        protected int CurrentEpoch;
        protected DelegateforWeights regulatorW;
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
            currentNN.Reg = new DefaultRegularization();
            regularization = new DefaultRegularization();
            regulatorW = (double d) => { return 0; };
            if (Optimizer.regulList != null && Optimizer.regulList.Count != 0)
            {
                Regularization regu = new Regularization();
                int IsComb = 0;
                for (int i = 0; i < Optimizer.regulList.Count; i++)
                {
                    Regularizations reg = Optimizer.regulList[i];
                    switch (reg)
                    {
                        case Regularizations.L1:
                            regulatorW = regu.L1;
                            IsComb++;
                            break;
                        case Regularizations.L2:
                            regulatorW = regu.L2;
                            IsComb++;
                            break;
                        case Regularizations.Dropout:
                            regularization = new RegDropoutLearn();
                            currentNN.Reg = new RegDropoutTest();
                            break;
                            /*case Regularizations.BatchNormalization:
                                regularization = new BatchNormalizationLearn();
                                currentNN.Reg
                                break;
                            */
                    }
                }
                if (IsComb == 2)
                {
                    regulatorW = regu.L1andL2;
                }
            }
        }
        public abstract double Backpropagation<T>(T dataset, int[] expected, int epoch);
        public virtual void Backpropagation(double learningRate)
        {
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                Optimizer.BackPropagation(currentNeuron, outputs, learningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    Optimizer.BackPropagation(currentNeuron, outputs, learningRate, CurrentEpoch, regulatorW);
                }
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

        protected double GetActual(double[] outputs, int expected)
        {
            double actual = regularization.FeedForward(outputs, currentNN.neuronLayer);
            return actual;
        }
    }

    public class GradientDescent : AbGradientDescent
    {
        public GradientDescent(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                double metr = 0;
                for (int i = 0; i < epoch; i++)
                {
                    UniqueRandom uniqueRandom = new UniqueRandom(0, expected.Length - 1);
                    CurrentEpoch++;
                    List<double> deltas;
                    for (int j = 0; j < expected.Length; j++)
                    {
                        int numb = uniqueRandom.GetUniqueRandomNumber();
                        int exp = expected[numb];
                        double actual = GetActual(outputsM[numb], exp);
                        double error = actual - exp;
                        metr += metric.GetMetric(actual, exp);
                        deltas = Deltapropagation(error);

                    }
                }
                return metr / epoch * 100;
            }
            else
            {
                return 0;
            }
        }
    }

    public class MiniBatchGD : AbGradientDescent
    {
        Random random;
        int Capacity;
        public MiniBatchGD(NeuralNetwork nn, int capacity = 2, IOptimizer optimizer = null) : base(nn, optimizer) 
        {
            if (capacity < 1)
            {
                return;
            }
            Capacity = capacity; 
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                double MSE = 0;
                for (int j = 0; j < epoch; j++)
                {
                    double MSEtemp = 0;
                    double totalError = 0;
                    CurrentEpoch++;
                    MSEtemp = MSEtemp / Capacity;
                    MSE += MSEtemp;
                    totalError = totalError / Capacity;
                    Backpropagation(currentNN.Topology.LearningRate);
                }
                return MSE / epoch * 100;
            }
            else
            {
                return 0;
            }
        }
    }

    public class StochasticGradient : AbGradientDescent
    {
        Random random;
        public StochasticGradient(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is double[][] outputsM)
            {
                double metr = 0;
                for (int i = 0; i < epoch; i++)
                {
                    UniqueRandom uniqueRandom = new UniqueRandom(0, expected.Length - 1);
                    CurrentEpoch++;
                    for (int j = 0; j < expected.Length; j++)
                    {
                        int numb = uniqueRandom.GetUniqueRandomNumber();
                        int exp = expected[numb];
                        double actual = GetActual(outputsM[numb], exp);
                        double error = actual - exp;
                        metr += metric.GetMetric(actual, exp);
                        Deltapropagation(error);
                        Backpropagation(topology.LearningRate);
                    }
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
