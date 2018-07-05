# asdnet
This is the pytorch code of the proposed asdnet for medical image segmentation


'''
GDL loss for the image reconstruction
By Dong Nie
'''
class gdl_loss(nn.Module):
    def __init__(self):  
        super(gdl_loss, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1,2), stride=1, padding=(0,1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2,1), stride=1, padding=(1,0), bias=False)
        
        filterX = torch.FloatTensor([[-1,1]]) # 1x2
        filterY = torch.FloatTensor([[1],[-1]]) # 2x1
        self.convX.weight = torch.nn.Parameter(filterX)
        self.convY.weight = torch.nn.parameter(filterY)
        
    def forward(self, pred, gt, pnorm=2):
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())

        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))
        
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        
        mat_loss = grad_diff_x ** pnorm + grad_diff_y ** pnorm # Batch x Channel x width x height
        
        shape = gt.shape
        
        mean_loss = torch.sum(mat_loss)/(shape[0]*shape[1]*shape[2]*shape[3])
        
        return mean_loss
