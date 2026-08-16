import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from avalanche.training.plugins import SupervisedPlugin
from torchvision import transforms

class IncrementalScalerTransform:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mean_ = None
        self.seen_ = 0
    
    def __call__(self, x):
        if not hasattr(self.scaler, 'mean_'):
            return x
        
        orig_device = x.device
        orig_dtype = x.dtype
        orig_shape = x.shape
        
        x_np = x.cpu().numpy().reshape(1, -1)
        x_norm = self.scaler.transform(x_np)
        
        # ✅ CORREÇÃO: atribuir o resultado do reshape
        x_normalized = torch.from_numpy(x_norm).to(orig_device).to(orig_dtype)
        x_normalized = x_normalized.reshape(orig_shape)  # Atribui o resultado
        
        return x_normalized
        
    def transform_batch(self, x_tensor):
        """Aplica a normalização no batch completo vindo da estratégia."""
        if not hasattr(self.scaler, 'mean_'):
            return x_tensor
            
        orig_device = x_tensor.device
        orig_dtype = x_tensor.dtype
        orig_shape = x_tensor.shape #[cite: 506, 507, 508]
        
        # Converte para numpy e achata se necessário [cite: 510]
        x_np = x_tensor.cpu().numpy()
        if len(orig_shape) > 2:
            x_np = x_np.reshape(orig_shape[0], -1)
            
        x_norm = self.scaler.transform(x_np) #[cite: 513]
        
        # Volta para Tensor e restaura o shape original [cite: 515, 518]
        return torch.from_numpy(x_norm).to(orig_device).to(orig_dtype).view(orig_shape)
    
    
    def update(self, x):
        """
        x: Deve ser o lote completo da experiência (ou um grande conjunto)
        para garantir estabilidade estatística.
        """
        # Se x for Tensor, converte para numpy
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Garante que seja 2D para o sklearn
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        elif len(x.shape) == 1:
            x = x.reshape(1, -1)

        # print('tamanho amostras transform', len(x))
        self.scaler.partial_fit(x)
        self.mean_ = self.scaler.mean_
        self.seen_ = self.scaler.n_samples_seen_
        


class UpdateScalerPlugin(SupervisedPlugin):
    def __init__(self, scaler_transform):
        """
        :param scaler_transform: Instância da classe IncrementalScalerTransform
                                 que está dentro do seu transform_groups.
        """
        super().__init__()
        self.scaler_transform = scaler_transform
        self._to_tensor = transforms.ToTensor()


    def before_training_exp(self, strategy, **kwargs):
        """
        Este gancho é chamado antes de cada experiência de treino.
        """
        # 1. Acessamos o dataset da experiência atual.
        # .replace_current_transform_group(None) é vital para pegar os dados BRUTOS
        # sem aplicar a normalização antiga neles antes de calcular a nova média.
        # raw_dataset = strategy.experience.dataset.replace_current_transform_group(None)

        # # 2. Carregamos os dados (x) da experiência.
        # # Se o dataset for muito grande, você pode iterar em mini-batches aqui,
        # # mas para 5 usuários de sensores, geralmente cabe na memória:
        # # print('raw_dataset', transforms.ToTensor(raw_dataset[:][0][0]))
        # # raw_x = self._to_tensor(raw_dataset[:][0][0])
        
        # data = [x[0] for x in raw_dataset]
        # print('len', len(data))

        # # print(data)
        # raw_x = np.stack(data)
        
        # # print('raw_x', np.sort(raw_x.flatten())[:10])#raw_x[:10])

        # # 3. Atualizamos o scaler com os dados brutos da nova experiência
        # # O método .update() que criamos já lida com a conversão Tensor -> NumPy
        # self.scaler_transform.update(raw_x)
        # print('mean_', self.scaler_transform.mean_)
        # print('seen_', self.scaler_transform.seen_)
        # print('scale_', self.scaler_transform.scale_)
        # print('var_', self.scaler_transform.var_)
        # print('n_features_in_', self.scaler_transform.n_features_in_)
        # print('feature_names_in_', self.scaler_transform.feature_names_in_)
        # print('mean', self.scaler_transform.mean_[:10])
        # print('seen', self.scaler_transform.seen_)

        # print(f"-> Estatísticas do Scaler atualizadas para a experiência {strategy.experience.current_experience}")

    def before_training_iteration(self, strategy, **kwargs):
        """Normaliza o mini-batch (MB) completo antes do forward pass."""
        # strategy.mb_x contém o batch atual de treino
        # print(f'mb_x before {strategy.experience.current_experience}', 
        #       np.sort(strategy.mb_x.flatten())[:10])
        raw_dataset = strategy.mb_x#strategy.experience.dataset.replace_current_transform_group(None)
        # print(raw_dataset.size(), raw_dataset)
        # 2. Carregamos os dados (x) da experiência.
        # Se o dataset for muito grande, você pode iterar em mini-batches aqui,
        # mas para 5 usuários de sensores, geralmente cabe na memória:
        # print('raw_dataset', transforms.ToTensor(raw_dataset[:][0][0]))
        # raw_x = self._to_tensor(raw_dataset[:][0][0])
        
        # data = [x[0].cpu().numpy() for x in raw_dataset]
        data = raw_dataset.cpu().numpy()
        # print('len', len(data), data)

        # # print(data)
        # raw_x = np.stack(data)
        
        # print('raw_x', np.sort(raw_x.flatten())[:10])#raw_x[:10])

        # 3. Atualizamos o scaler com os dados brutos da nova experiência
        # O método .update() que criamos já lida com a conversão Tensor -> NumPy
        self.scaler_transform.update(strategy.mb_x)
        x_norm = self.scaler_transform.transform_batch(strategy.mb_x)
        strategy.mb_x.copy_(x_norm)
        # print(f'mb_x after {strategy.experience.current_experience}', 
        #       np.sort(strategy.mb_x.flatten())[:10])
        
    def before_eval_iteration(self, strategy, **kwargs):
        """Garante que os dados de teste/validação também sejam normalizados."""
        x_norm = self.scaler_transform.transform_batch(strategy.mb_x)
        strategy.mb_x.copy_(x_norm) 