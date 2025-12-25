# final
class GLIM(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=1, vert_anchors=10, horz_anchors=10,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.block_size = 10
        # transformer
        self.mamba_blocks_global = nn.Sequential(*[VSSBlock(hidden_dim=self.n_embd, drop_path=0.1)
                                                   for layer in range(n_layer)])
        # transformer
        self.mamba_blocks_local = nn.Sequential(*[VSSBlock(hidden_dim=self.n_embd, drop_path=0.1)
                                                  for layer in range(n_layer)])
        # transformer
        self.mamba_blocks_c1 = nn.Sequential(*[VSSBlock(hidden_dim=d_model , drop_path=0.1)
                                               for layer in range(n_layer)])


        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        if self.n_embd == 256:
            self.avgpool_local = nn.AdaptiveAvgPool2d((80, 80))

        elif self.n_embd == 512:
            self.avgpool_local = nn.AdaptiveAvgPool2d((40, 40))

        elif self.n_embd == 1024:
            self.avgpool_local = nn.AdaptiveAvgPool2d((20, 20))

    def forward(self, x):
        # """
        # Args:
        #     x (tuple?)
        #
        # """
        # # -------------------------------------------------------------------------
        # # Configure
        # # -------------------------------------------------------------------------
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        xxx = torch.concat((rgb_fea, ir_fea), dim=0)
        xxx = self.avgpool_local(xxx)
        _, c, h1, w1 = xxx.shape
        h_unfold = h1 // 10
        w_unfold = w1 // 10
        # -------------------------------------------------------------------------
        # Local
        # -------------------------------------------------------------------------
        xxx = xxx.unfold(2, 10, 10).unfold(3, 10, 10)  # [b,c,hu,wu,bs,bs]
        xxx = xxx.permute(0, 2, 3, 1, 4, 5).contiguous()  # [b,hu,wu,c,bs,bs]
        xxx = xxx.view(-1, c, 10, 10)  # [b*hu*wu, c, bs, bs]
        xxx = xxx.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        xxx = self.mamba_blocks_local(xxx)  # dim:(B, 2*H*W, C)
        # decoder head
        xxx = self.ln_f(xxx)  # dim:(B, 2*H*W, C)

        # 步骤2: 恢复分块维度 → [b, hu, wu, c, 10, 10]
        xxx = xxx.view(bs*2, h_unfold, w_unfold, c, 10, 10)

        # 步骤3: 调整维度顺序 → [b, c, hu, wu, 10, 10]
        xxx = xxx.permute(0, 3, 1, 2, 4, 5)

        # 步骤4: 折叠分块 → [b, c, hu*10, wu*10]
        xxx = xxx.permute(0, 1, 2, 4, 3, 5).contiguous().view(bs*2, c, h_unfold * 10, w_unfold * 10)
        rgb_fea_local, ir_fea_local = torch.split(xxx, [bs, bs], dim=0)
        #
        xx = torch.concat((rgb_fea,ir_fea),dim=0)
        xx = self.avgpool(xx)
        # -------------------------------------------------------------------------
        # Mamba b h w d
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        xx = xx.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        # Mamba
        xx = self.mamba_blocks_global(xx)  # dim:(B, 2*H*W, C)
        # decoder head
        xx = self.ln_f(xx)  # dim:(B, 2*H*W, C)
        xx = xx.view(bs*2, 10, 10, c)
        xx = xx.permute(0, 3, 1, 2)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        xx = xx[:, :, :, :].contiguous().view(bs*2, c, 10, 10)
        rgb_fea_global, ir_fea_global = torch.split(xx, [bs, bs], dim=0)

        rgb_fea_global = F.interpolate(rgb_fea_global, size=([h, w]), mode='bilinear')
        ir_fea_global = F.interpolate(ir_fea_global, size=([h, w]), mode='bilinear')

        rgb_fea = torch.concat((rgb_fea_global, rgb_fea_local), dim=1)
        ir_fea = torch.concat((ir_fea_global, ir_fea_local), dim=1)

        x_rgb = rgb_fea
        x_ir = ir_fea

        ir_sp = x_rgb - x_ir
        rgb_sp = x_ir - x_rgb
        x_an = x_ir + x_rgb

        x = torch.concat((ir_sp,rgb_sp,x_an),dim=0)
        x = x.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        x = self.mamba_blocks_c1(x)  # dim:(B, 2*H*W, C)
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs*3, h, w, c)
        x = x.permute(0, 3, 1, 2)
        x = x[:, :, :, :].contiguous().view(bs*3, c, h, w)
        ir_sp, rgb_sp, x_an = torch.split(x, [bs, bs, bs], dim=0)

        rgb_fea_out = x_an + rgb_sp
        ir_fea_out = x_an + ir_sp

        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out, ir_sp, 0, 0, 0