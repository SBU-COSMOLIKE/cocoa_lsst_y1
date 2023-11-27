# LSST-Y1 COLA Chains
This branch of the `lsst_y1` repository contains yaml files for LSST-Y1 chains to be run in Cocoa, using the COLA emulators as well as Euclid Emulator 2. **Please, read this documentation before running any chains.**

**If you encounter any bugs while running chains, please report!**

## COLA Emulators
The source code for all COLA emulators used in the project are stored under the `COLA_Emulators/` folder.

## Analysis Description
The fiducial data vector `data/EE2_FIDUCIAL_NO_NOISE.modelvector` was generated with the yaml file `yamls/CREATE_EE2_FIDUCIAL_DATA_VECTOR.yaml`. A noise realization was added to the data vector, using the Python script `data/add_noise_to_data_vector.py`, resulting in the file `data/EE2_FIDUCIAL_NOISED.modelvector`. For each of the masks, one `dataset` file was created, in `data/LSST_Y1_MX_EE2_FID.dataset`, where `X` represents the number of tha mask, `M1` being the most aggressive and `M5` the most conservative. For testing purposes, an additional dataset `LSST_Y1_M6_EE2_NO_NOISE.dataset` was created using the noiseless data vector. The noiseless dataset is used in `TEST_COLA_EMU_SHEAR.yaml`, which is a simple evaluation at the fiducial parameters.

Using the same methodology, other eight fiducial data vectors were created, varying the parameters `Omega_m`, `As` and `ns`. We define a 2x2 grid in the `Omega_m`-`As` plane, and choose values in-between the reference and borders of each parameter limit. For instance, the parameter `As` has limits `1.7e-9` to `2.5e-9`, and the reference is in the middle, `2.1e-9`. Thus, we choose two values of `As`: the low value `1.9e-9` and high value `2.3e-9`. `Omega_m` varies to `0.28` and `0.36`, and `ns` varies to `0.94` and `0.98`. The yamls used to create the eight data vectors are saved in `yamls/OM_[HIGH/LOW]_[AS-NS]_[HIGH/LOW].yaml`. After that, noise realization was added to each data vector.

All chains run with LSST-Y1 Cosmic Shear, same nuisance parameter priors (pessimistic IA with NLA, optimistic photo-z and shear calibration). The priors for cosmological parameters are the EE2 box boundaries. We are only considering wCDM. The convergence criterion is given by `R-1 < 0.005`.

The variables between chains are: masks (scale cuts), the emulators (EE2, Halofit, COLA), the number of anchors (1/10/25/100) and the fiducial data vector (EE2 reference, Omega_m high/low, As high/low, ns high/low).

The nonlinear emulators are set in the yaml file under the `non_linear_emul` option. They are numbered as:

1 - EE2;

2 - Halofit;

3 - GP;

4 - NN;

5 - PCE;

For COLA Emulators, the number of anchors is set in the yaml file under the option `num_refs`, set to 1 by default.

To change between LCDM and wCDM COLA emulators, use the option `cola_emu_mode: LCDM` or `cola_emu_mode: wCDM`.

More emulators will be added since we need to implement the different anchors, COLA precision, dark energy models...

**NOTE**: There are still some placeholder strings in the yaml files, namely `COLA_EMU_NO_ANCHOR`, `COLA_EMU_10_ANCHORS`, `COLA_EMU_25_ANCHORS` and `COLA_EMU_100_ANCHORS`. This is because we haven't decided on the emulator to run the bulk of the chains and the emulators with multiple anchors are still not implemented. Before running chains, the placeholder strings need to be substituted by the actual numbers representing the chosen emulator.

## List of chains
The format for each entry is

INDEX - Emulator / Mask / Model / Anchors / Fiducial cosmology

1 - EE2 / M1 / wCDM / -- / EE2 reference

2 - EE2 / M2 / wCDM / -- / EE2 reference

3 - EE2 / M3 / wCDM / -- / EE2 reference

4 - EE2 / M4 / wCDM / -- / EE2 reference

5 - EE2 / M5 / wCDM / -- / EE2 reference

6 - COLA / M1 / wCDM / 1 anchor / EE2 reference

7 - COLA / M2 / wCDM / 1 anchor / EE2 reference

8 - COLA / M3 / wCDM / 1 anchor / EE2 reference

9 - COLA / M4 / wCDM / 1 anchor / EE2 reference

10 - COLA / M5 / wCDM / 1 anchor / EE2 reference

11 - COLA / M1 / wCDM / 10 anchors / EE2 reference

12 - COLA / M2 / wCDM / 10 anchors / EE2 reference

13 - COLA / M3 / wCDM / 10 anchors / EE2 reference

14 - COLA / M4 / wCDM / 10 anchors / EE2 reference

15 - COLA / M5 / wCDM / 10 anchors / EE2 reference

16 - COLA / M1 / wCDM / 25 anchors / EE2 reference

17 - COLA / M2 / wCDM / 25 anchors / EE2 reference

18 - COLA / M3 / wCDM / 25 anchors / EE2 reference

19 - COLA / M4 / wCDM / 25 anchors / EE2 reference

20 - COLA / M5 / wCDM / 25 anchors / EE2 reference

21 - COLA / M1 / wCDM / 100 anchors / EE2 reference

22 - COLA / M2 / wCDM / 100 anchors / EE2 reference

23 - COLA / M3 / wCDM / 100 anchors / EE2 reference

24 - COLA / M4 / wCDM / 100 anchors / EE2 reference

25 - COLA / M5 / wCDM / 100 anchors / EE2 reference

26 - Halofit / M1 / wCDM / -- / EE2 reference

27 - Halofit / M2 / wCDM / -- / EE2 reference

28 - Halofit / M3 / wCDM / -- / EE2 reference

29 - Halofit / M4 / wCDM / -- / EE2 reference

30 - Halofit / M5 / wCDM / -- / EE2 reference

31 - EE2 / M1 / wCDM / -- / High Omega_m High As

32 - EE2 / M2 / wCDM / -- / High Omega_m High As

33 - EE2 / M3 / wCDM / -- / High Omega_m High As

34 - COLA / M1 / wCDM / 1 anchor / High Omega_m High As

35 - COLA / M2 / wCDM / 1 anchor / High Omega_m High As

36 - COLA / M3 / wCDM / 1 anchor / High Omega_m High As

37 - COLA / M1 / wCDM / 10 anchors / High Omega_m High As

38 - COLA / M2 / wCDM / 10 anchors / High Omega_m High As

39 - COLA / M3 / wCDM / 10 anchors / High Omega_m High As

40 - COLA / M1 / wCDM / 25 anchors / High Omega_m High As

41 - COLA / M2 / wCDM / 25 anchors / High Omega_m High As

42 - COLA / M3 / wCDM / 25 anchors / High Omega_m High As

43 - COLA / M1 / wCDM / 100 anchors / High Omega_m High As

44 - COLA / M2 / wCDM / 100 anchors / High Omega_m High As

45 - COLA / M3 / wCDM / 100 anchors / High Omega_m High As

46 - EE2 / M1 / wCDM / -- / Low Omega_m High As

47 - EE2 / M2 / wCDM / -- / Low Omega_m High As

48 - EE2 / M3 / wCDM / -- / Low Omega_m High As

49 - COLA / M1 / wCDM / 1 anchor / Low Omega_m High As

50 - COLA / M2 / wCDM / 1 anchor / Low Omega_m High As

51 - COLA / M3 / wCDM / 1 anchor / Low Omega_m High As

52 - COLA / M1 / wCDM / 10 anchors / Low Omega_m High As

53 - COLA / M2 / wCDM / 10 anchors / Low Omega_m High As

54 - COLA / M3 / wCDM / 10 anchors / Low Omega_m High As

55 - COLA / M1 / wCDM / 25 anchors / Low Omega_m High As

56 - COLA / M2 / wCDM / 25 anchors / Low Omega_m High As

57 - COLA / M3 / wCDM / 25 anchors / Low Omega_m High As

58 - COLA / M1 / wCDM / 100 anchors / Low Omega_m High As

59 - COLA / M2 / wCDM / 100 anchors / Low Omega_m High As

60 - COLA / M3 / wCDM / 100 anchors / Low Omega_m High As

61 - EE2 / M1 / wCDM / -- / High Omega_m High ns

62 - EE2 / M2 / wCDM / -- / High Omega_m High s

63 - EE2 / M3 / wCDM / -- / High Omega_m High ns

64 - COLA / M1 / wCDM / 1 anchor / High Omega_m High ns

65 - COLA / M2 / wCDM / 1 anchor / High Omega_m High ns

66 - COLA / M3 / wCDM / 1 anchor / High Omega_m High ns

67 - COLA / M1 / wCDM / 10 anchors / High Omega_m High ns

68 - COLA / M2 / wCDM / 10 anchors / High Omega_m High ns

69 - COLA / M3 / wCDM / 10 anchors / High Omega_m High ns

70 - COLA / M1 / wCDM / 25 anchors / High Omega_m High ns

71 - COLA / M2 / wCDM / 25 anchors / High Omega_m High ns

72 - COLA / M3 / wCDM / 25 anchors / High Omega_m High ns

73 - COLA / M1 / wCDM / 100 anchors / High Omega_m High ns

74 - COLA / M2 / wCDM / 100 anchors / High Omega_m High ns

75 - COLA / M3 / wCDM / 100 anchors / High Omega_m High ns

76 - EE2 / M1 / wCDM / -- / Low Omega_m High ns

77 - EE2 / M2 / wCDM / -- / Low Omega_m High ns

78 - EE2 / M3 / wCDM / -- / Low Omega_m High ns

79 - COLA / M1 / wCDM / 1 anchor / Low Omega_m High ns

80 - COLA / M2 / wCDM / 1 anchor / Low Omega_m High ns

81 - COLA / M3 / wCDM / 1 anchor / Low Omega_m High ns

82 - COLA / M1 / wCDM / 10 anchors / Low Omega_m High ns

83 - COLA / M2 / wCDM / 10 anchors / Low Omega_m High ns

84 - COLA / M3 / wCDM / 10 anchors / Low Omega_m High ns

85 - COLA / M1 / wCDM / 25 anchors / Low Omega_m High ns

86 - COLA / M2 / wCDM / 25 anchors / Low Omega_m High ns

87 - COLA / M3 / wCDM / 25 anchors / Low Omega_m High ns

88 - COLA / M1 / wCDM / 100 anchors / Low Omega_m High ns

89 - COLA / M2 / wCDM / 100 anchors / Low Omega_m High ns

90 - COLA / M3 / wCDM / 100 anchors / Low Omega_m High ns

91 - EE2 / M1 / wCDM / -- / High Omega_m Low As

92 - EE2 / M2 / wCDM / -- / High Omega_m Low As

93 - EE2 / M3 / wCDM / -- / High Omega_m Low As

94 - COLA / M1 / wCDM / 1 anchor / High Omega_m Low As

95 - COLA / M2 / wCDM / 1 anchor / High Omega_m Low As

96 - COLA / M3 / wCDM / 1 anchor / High Omega_m Low As

97 - COLA / M1 / wCDM / 10 anchors / High Omega_m Low As

98 - COLA / M2 / wCDM / 10 anchors / High Omega_m Low As

99 - COLA / M3 / wCDM / 10 anchors / High Omega_m Low As

100 - COLA / M1 / wCDM / 25 anchors / High Omega_m Low As

101 - COLA / M2 / wCDM / 25 anchors / High Omega_m Low As

102 - COLA / M3 / wCDM / 25 anchors / High Omega_m Low As

103 - COLA / M1 / wCDM / 100 anchors / High Omega_m Low As

104 - COLA / M2 / wCDM / 100 anchors / High Omega_m Low As

105 - COLA / M3 / wCDM / 100 anchors / High Omega_m Low As

106 - EE2 / M1 / wCDM / -- / Low Omega_m Low As

107 - EE2 / M2 / wCDM / -- / Low Omega_m Low As

108 - EE2 / M3 / wCDM / -- / Low Omega_m Low As

109 - COLA / M1 / wCDM / 1 anchor / Low Omega_m Low As

110 - COLA / M2 / wCDM / 1 anchor / Low Omega_m Low As

111 - COLA / M3 / wCDM / 1 anchor / Low Omega_m Low As

112 - COLA / M1 / wCDM / 10 anchors / Low Omega_m Low As

113 - COLA / M2 / wCDM / 10 anchors / Low Omega_m Low As

114 - COLA / M3 / wCDM / 10 anchors / Low Omega_m Low As

115 - COLA / M1 / wCDM / 25 anchors / Low Omega_m Low As

116 - COLA / M2 / wCDM / 25 anchors / Low Omega_m Low As

117 - COLA / M3 / wCDM / 25 anchors / Low Omega_m Low As

118 - COLA / M1 / wCDM / 100 anchors / Low Omega_m Low As

119 - COLA / M2 / wCDM / 100 anchors / Low Omega_m Low As

120 - COLA / M3 / wCDM / 100 anchors / Low Omega_m Low As

121 - EE2 / M1 / wCDM / -- / High Omega_m Low ns

122 - EE2 / M2 / wCDM / -- / High Omega_m Low ns

123 - EE2 / M3 / wCDM / -- / High Omega_m Low ns

124 - COLA / M1 / wCDM / 1 anchor / High Omega_m Low ns

125 - COLA / M2 / wCDM / 1 anchor / High Omega_m Low ns

126 - COLA / M3 / wCDM / 1 anchor / High Omega_m Low ns

127 - COLA / M1 / wCDM / 10 anchors / High Omega_m Low ns

128 - COLA / M2 / wCDM / 10 anchors / High Omega_m Low ns

129 - COLA / M3 / wCDM / 10 anchors / High Omega_m Low ns

130 - COLA / M1 / wCDM / 25 anchors / High Omega_m Low ns

131 - COLA / M2 / wCDM / 25 anchors / High Omega_m Low ns

132 - COLA / M3 / wCDM / 25 anchors / High Omega_m Low ns

133 - COLA / M1 / wCDM / 100 anchors / High Omega_m Low ns

134 - COLA / M2 / wCDM / 100 anchors / High Omega_m Low ns

135 - COLA / M3 / wCDM / 100 anchors / High Omega_m Low ns

136 - EE2 / M1 / wCDM / -- / Low Omega_m Low ns

137 - EE2 / M2 / wCDM / -- / Low Omega_m Low ns

138 - EE2 / M3 / wCDM / -- / Low Omega_m Low ns

139 - COLA / M1 / wCDM / 1 anchor / Low Omega_m Low ns

140 - COLA / M2 / wCDM / 1 anchor / Low Omega_m Low ns

141 - COLA / M3 / wCDM / 1 anchor / Low Omega_m Low ns

142 - COLA / M1 / wCDM / 10 anchors / Low Omega_m Low ns

143 - COLA / M2 / wCDM / 10 anchors / Low Omega_m Low ns

144 - COLA / M3 / wCDM / 10 anchors / Low Omega_m Low ns

145 - COLA / M1 / wCDM / 25 anchors / Low Omega_m Low ns

146 - COLA / M2 / wCDM / 25 anchors / Low Omega_m Low ns

147 - COLA / M3 / wCDM / 25 anchors / Low Omega_m Low ns

148 - COLA / M1 / wCDM / 100 anchors / Low Omega_m Low ns

149 - COLA / M2 / wCDM / 100 anchors / Low Omega_m Low ns

150 - COLA / M3 / wCDM / 100 anchors / Low Omega_m Low ns

151 - EE2 / M5 / wCDM / -- / EE2 reference

152 - COLA / M5 / wCDM / 1 anchor / EE2 reference

153 - COLA / M5 / wCDM / 25 anchors / EE2 reference

154 - EE2 / M5 / wCDM / -- / High Omega_m High As

155 - COLA / M5 / wCDM / 1 anchor / High Omega_m High As

156 - COLA / M5 / wCDM / 25 anchors / High Omega_m High As

157 - EE2 / M5 / wCDM / -- / Low Omega_m High As

158 - COLA / M5 / wCDM / 1 anchor / Low Omega_m High As

159 - COLA / M5 / wCDM / 25 anchors / Low Omega_m High As

160 - EE2 / M5 / wCDM / -- / High Omega_m High ns

161 - COLA / M5 / wCDM / 1 anchor / High Omega_m High ns

162 - COLA / M5 / wCDM / 25 anchors / High Omega_m High ns

163 - EE2 / M5 / wCDM / -- / Low Omega_m High ns

164 - COLA / M5 / wCDM / 1 anchor / Low Omega_m High ns

165 - COLA / M5 / wCDM / 25 anchors / Low Omega_m High ns

166 - EE2 / M5 / wCDM / -- / High Omega_m Low As

167 - COLA / M5 / wCDM / 1 anchor / High Omega_m Low As

168 - COLA / M5 / wCDM / 25 anchors / High Omega_m Low As

169 - EE2 / M5 / wCDM / -- / Low Omega_m Low As

170 - COLA / M5 / wCDM / 1 anchor / Low Omega_m Low As

171 - COLA / M5 / wCDM / 25 anchors / Low Omega_m Low As

172 - EE2 / M5 / wCDM / -- / High Omega_m Low ns

173 - COLA / M5 / wCDM / 1 anchor / High Omega_m Low ns

174 - COLA / M5 / wCDM / 25 anchors / High Omega_m Low ns

175 - EE2 / M5 / wCDM / -- / Low Omega_m Low ns

176 - COLA / M5 / wCDM / 1 anchor / Low Omega_m Low ns

177 - COLA / M5 / wCDM / 25 anchors / Low Omega_m Low ns

178 - EE2 / M4 / wCDM / -- / EE2 reference

179 - COLA / M4 / wCDM / 1 anchor / EE2 reference

180 - COLA / M4 / wCDM / 25 anchors / EE2 reference

181 - EE2 / M4 / wCDM / -- / High Omega_m High As

182 - COLA / M4 / wCDM / 1 anchor / High Omega_m High As

183 - COLA / M4 / wCDM / 25 anchors / High Omega_m High As

184 - EE2 / M4 / wCDM / -- / Low Omega_m High As

185 - COLA / M4 / wCDM / 1 anchor / Low Omega_m High As

186 - COLA / M4 / wCDM / 25 anchors / Low Omega_m High As

187 - EE2 / M4 / wCDM / -- / High Omega_m High ns

188 - COLA / M4 / wCDM / 1 anchor / High Omega_m High ns

189 - COLA / M4 / wCDM / 25 anchors / High Omega_m High ns

190 - EE2 / M4 / wCDM / -- / Low Omega_m High ns

191 - COLA / M4 / wCDM / 1 anchor / Low Omega_m High ns

192 - COLA / M4 / wCDM / 25 anchors / Low Omega_m High ns

193 - EE2 / M4 / wCDM / -- / High Omega_m Low As

194 - COLA / M4 / wCDM / 1 anchor / High Omega_m Low As

195 - COLA / M4 / wCDM / 25 anchors / High Omega_m Low As

196 - EE2 / M4 / wCDM / -- / Low Omega_m Low As

197 - COLA / M4 / wCDM / 1 anchor / Low Omega_m Low As

198 - COLA / M4 / wCDM / 25 anchors / Low Omega_m Low As

199 - EE2 / M4 / wCDM / -- / High Omega_m Low ns

200 - COLA / M4 / wCDM / 1 anchor / High Omega_m Low ns

201 - COLA / M4 / wCDM / 25 anchors / High Omega_m Low ns

202 - EE2 / M4 / wCDM / -- / Low Omega_m Low ns

203 - COLA / M4 / wCDM / 1 anchor / Low Omega_m Low ns

204 - COLA / M4 / wCDM / 25 anchors / Low Omega_m Low ns

## LCDM chains

205 - EE2 / M2 / LCDM / -- / EE2 ref

206 - COLA / M2 / LCDM / 1 anchor / EE2 ref

207 - COLA / M2 / LCDM / 25 anchors / EE2 ref

208 - EE2 / M2 / LCDM / -- / High Omega_m High As

209 - COLA / M2 / LCDM / 1 anchor / High Omega_m High As

210 - COLA / M2 / LCDM / 25 anchors / High Omega_m High As

211 - EE2 / M2 / LCDM / -- / Low Omega_m High As

212 - COLA / M2 / LCDM / 1 anchor / Low Omega_m High As

213 - COLA / M2 / LCDM / 25 anchors / Low Omega_m High As

214 - EE2 / M2 / LCDM / -- / High Omega_m Low As

215 - COLA / M2 / LCDM / 1 anchor / High Omega_m Low As

216 - COLA / M2 / LCDM / 25 anchors / High Omega_m Low As

217 - EE2 / M2 / LCDM / -- / Low Omega_m Low As

218 - COLA / M2 / LCDM / 1 anchor / Low Omega_m Low As

219 - COLA / M2 / LCDM / 25 anchors / Low Omega_m Low As

220 - EE2 / M2 / LCDM / -- / High Omega_m High ns

221 - COLA / M2 / LCDM / 1 anchor / High Omega_m High ns

222 - COLA / M2 / LCDM / 25 anchors / High Omega_m High ns

223 - EE2 / M2 / LCDM / -- / Low Omega_m High ns

224 - COLA / M2 / LCDM / 1 anchor / Low Omega_m High ns

225 - COLA / M2 / LCDM / 25 anchors / Low Omega_m High ns

226 - EE2 / M2 / LCDM / -- / High Omega_m Low ns

227 - COLA / M2 / LCDM / 1 anchor / High Omega_m Low ns

228 - COLA / M2 / LCDM / 25 anchors / High Omega_m Low ns

229 - EE2 / M2 / LCDM / -- / Low Omega_m Low ns

230 - COLA / M2 / LCDM / 1 anchor / Low Omega_m Low ns

231 - COLA / M2 / LCDM / 25 anchors / Low Omega_m Low ns

232 - EE2 / M3 / LCDM / -- / EE2 ref

233 - COLA / M3 / LCDM / 1 anchor / EE2 ref

234 - COLA / M3 / LCDM / 25 anchors / EE2 ref

235 - EE2 / M3 / LCDM / -- / High Omega_m High As

236 - COLA / M3 / LCDM / 1 anchor / High Omega_m High As

237 - COLA / M3 / LCDM / 25 anchors / High Omega_m High As

238 - EE2 / M3 / LCDM / -- / Low Omega_m High As

239 - COLA / M3 / LCDM / 1 anchor / Low Omega_m High As

240 - COLA / M3 / LCDM / 25 anchors / Low Omega_m High As

241 - EE2 / M3 / LCDM / -- / High Omega_m Low As

242 - COLA / M3 / LCDM / 1 anchor / High Omega_m Low As

243 - COLA / M3 / LCDM / 25 anchors / High Omega_m Low As

244 - EE2 / M3 / LCDM / -- / Low Omega_m Low As

245 - COLA / M3 / LCDM / 1 anchor / Low Omega_m Low As

246 - COLA / M3 / LCDM / 25 anchors / Low Omega_m Low As

247 - EE2 / M3 / LCDM / -- / High Omega_m High ns

248 - COLA / M3 / LCDM / 1 anchor / High Omega_m High ns

249 - COLA / M3 / LCDM / 25 anchors / High Omega_m High ns

250 - EE2 / M3 / LCDM / -- / Low Omega_m High ns

251 - COLA / M3 / LCDM / 1 anchor / Low Omega_m High ns

252 - COLA / M3 / LCDM / 25 anchors / Low Omega_m High ns

253 - EE2 / M3 / LCDM / -- / High Omega_m Low ns

254 - COLA / M3 / LCDM / 1 anchor / High Omega_m Low ns

255 - COLA / M3 / LCDM / 25 anchors / High Omega_m Low ns

256 - EE2 / M3 / LCDM / -- / Low Omega_m Low ns

257 - COLA / M3 / LCDM / 1 anchor / Low Omega_m Low ns

258 - COLA / M3 / LCDM / 25 anchors / Low Omega_m Low ns

259 - EE2 / M4 / LCDM / -- / EE2 ref

260 - COLA / M4 / LCDM / 1 anchor / EE2 ref

261 - COLA / M4 / LCDM / 25 anchors / EE2 ref

262 - EE2 / M4 / LCDM / -- / High Omega_m High As

263 - COLA / M4 / LCDM / 1 anchor / High Omega_m High As

264 - COLA / M4 / LCDM / 25 anchors / High Omega_m High As

265 - EE2 / M4 / LCDM / -- / Low Omega_m High As

266 - COLA / M4 / LCDM / 1 anchor / Low Omega_m High As

267 - COLA / M4 / LCDM / 25 anchors / Low Omega_m High As

268 - EE2 / M4 / LCDM / -- / High Omega_m Low As

269 - COLA / M4 / LCDM / 1 anchor / High Omega_m Low As

270 - COLA / M4 / LCDM / 25 anchors / High Omega_m Low As

271 - EE2 / M4 / LCDM / -- / Low Omega_m Low As

272 - COLA / M4 / LCDM / 1 anchor / Low Omega_m Low As

273 - COLA / M4 / LCDM / 25 anchors / Low Omega_m Low As

274 - EE2 / M4 / LCDM / -- / High Omega_m High ns

275 - COLA / M4 / LCDM / 1 anchor / High Omega_m High ns

276 - COLA / M4 / LCDM / 25 anchors / High Omega_m High ns

277 - EE2 / M4 / LCDM / -- / Low Omega_m High ns

278 - COLA / M4 / LCDM / 1 anchor / Low Omega_m High ns

279 - COLA / M4 / LCDM / 25 anchors / Low Omega_m High ns

280 - EE2 / M4 / LCDM / -- / High Omega_m Low ns

281 - COLA / M4 / LCDM / 1 anchor / High Omega_m Low ns

282 - COLA / M4 / LCDM / 25 anchors / High Omega_m Low ns

283 - EE2 / M4 / LCDM / -- / Low Omega_m Low ns

284 - COLA / M4 / LCDM / 1 anchor / Low Omega_m Low ns

285 - COLA / M4 / LCDM / 25 anchors / Low Omega_m Low ns

## High precision wCDM chains: only M2, M3 and M4

286 - COLA high / M2 / wCDM / 1 anchor / EE2 ref

287 - COLA high / M2 / wCDM / 1 anchor / High Omega_m High As

288 - COLA high / M2 / wCDM / 1 anchor / High Omega_m Low As

289 - COLA high / M2 / wCDM / 1 anchor / Low Omega_m High As

290 - COLA high / M2 / wCDM / 1 anchor / Low Omega_m Low ns

291 - COLA high / M2 / wCDM / 1 anchor / High Omega_m High ns

292 - COLA high / M2 / wCDM / 1 anchor / High Omega_m Low ns

293 - COLA high / M2 / wCDM / 1 anchor / Low Omega_m High ns

294 - COLA high / M2 / wCDM / 1 anchor / Low Omega_m Low ns

295 - COLA high / M3 / wCDM / 1 anchor / EE2 ref

296 - COLA high / M3 / wCDM / 1 anchor / High Omega_m High As

297 - COLA high / M3 / wCDM / 1 anchor / High Omega_m Low As

298 - COLA high / M3 / wCDM / 1 anchor / Low Omega_m High As

299 - COLA high / M3 / wCDM / 1 anchor / Low Omega_m Low ns

300 - COLA high / M3 / wCDM / 1 anchor / High Omega_m High ns

301 - COLA high / M3 / wCDM / 1 anchor / High Omega_m Low ns

302 - COLA high / M3 / wCDM / 1 anchor / Low Omega_m High ns

303 - COLA high / M3 / wCDM / 1 anchor / Low Omega_m Low ns

304 - COLA high / M4 / wCDM / 1 anchor / EE2 ref

305 - COLA high / M4 / wCDM / 1 anchor / High Omega_m High As

306 - COLA high / M4 / wCDM / 1 anchor / High Omega_m Low As

307 - COLA high / M4 / wCDM / 1 anchor / Low Omega_m High As

308 - COLA high / M4 / wCDM / 1 anchor / Low Omega_m Low ns

309 - COLA high / M4 / wCDM / 1 anchor / High Omega_m High ns

310 - COLA high / M4 / wCDM / 1 anchor / High Omega_m Low ns

311 - COLA high / M4 / wCDM / 1 anchor / Low Omega_m High ns

312 - COLA high / M4 / wCDM / 1 anchor / Low Omega_m Low ns

## High precision LCDM chains: only M2, M3 and M4 (not written yet)

313 - COLA high / M2 / LCDM / 1 anchor / EE2 ref

314 - COLA high / M2 / LCDM / 1 anchor / High Omega_m High As

315 - COLA high / M2 / LCDM / 1 anchor / High Omega_m Low As

316 - COLA high / M2 / LCDM / 1 anchor / Low Omega_m High As

317 - COLA high / M2 / LCDM / 1 anchor / Low Omega_m Low ns

318 - COLA high / M2 / LCDM / 1 anchor / High Omega_m High ns

319 - COLA high / M2 / LCDM / 1 anchor / High Omega_m Low ns

320 - COLA high / M2 / LCDM / 1 anchor / Low Omega_m High ns

321 - COLA high / M2 / LCDM / 1 anchor / Low Omega_m Low ns

322 - COLA high / M3 / LCDM / 1 anchor / EE2 ref

323 - COLA high / M3 / LCDM / 1 anchor / High Omega_m High As

324 - COLA high / M3 / LCDM / 1 anchor / High Omega_m Low As

325 - COLA high / M3 / LCDM / 1 anchor / Low Omega_m High As

326 - COLA high / M3 / LCDM / 1 anchor / Low Omega_m Low ns

327 - COLA high / M3 / LCDM / 1 anchor / High Omega_m High ns

328 - COLA high / M3 / LCDM / 1 anchor / High Omega_m Low ns

329 - COLA high / M3 / LCDM / 1 anchor / Low Omega_m High ns

330 - COLA high / M3 / LCDM / 1 anchor / Low Omega_m Low ns

331 - COLA high / M4 / LCDM / 1 anchor / EE2 ref

332 - COLA high / M4 / LCDM / 1 anchor / High Omega_m High As

333 - COLA high / M4 / LCDM / 1 anchor / High Omega_m Low As

334 - COLA high / M4 / LCDM / 1 anchor / Low Omega_m High As

335 - COLA high / M4 / LCDM / 1 anchor / Low Omega_m Low ns

336 - COLA high / M4 / LCDM / 1 anchor / High Omega_m High ns

337 - COLA high / M4 / LCDM / 1 anchor / High Omega_m Low ns

338 - COLA high / M4 / LCDM / 1 anchor / Low Omega_m High ns

339 - COLA high / M4 / LCDM / 1 anchor / Low Omega_m Low ns

## Emulator comparison chains
The following chains were run first to assert equivalence between emulators and how well they perform against Halofit and EE2

900 - COLA GP / M1 / wCDM / 1 anchor / EE2 ref

901 - COLA NN / M1 / wCDM / 1 anchor / EE2 ref

902 - COLA PCE / M1 / wCDM / 1 anchor / EE2 ref

903 - COLA GP / M1 / LCDM / 1 anchor / EE2 ref

904 - COLA NN / M1 / LCDM / 1 anchor / EE2 ref

905 - COLA PCE / M1 / LCDM / 1 anchor / EE2 ref

906 - EE2 / M1 / LCDM / -- / EE2 ref

907 - Halofit / M1 / LCDM / -- / EE2 ref

908 - COLA GP / M1 / wCDM / 25 anchors / High Omega_m Low As

909 - COLA NN / M1 / wCDM / 25 anchors / High Omega_m Low As

910 - COLA PCE / M1 / wCDM / 25 anchors / High Omega_m Low As

911 - COLA PCE high / M2 / wCDM / 1 anchor / EE2 ref

912 - COLA PCE high / M2 / wCDM / 1 anchor / High Omega_m High As

913 - COLA PCE / M1 / LCDM / 1 anchor / High Omega_m High As

914 - COLA PCE+NN / M1 /  LCDM / 1 anchor / High Omega_m High As


## Running Cosmolike projects <a name="running_cosmolike_projects"></a> 

In this tutorial, we assume the user installed Cocoa via the *Conda installation* method, and the name of the Conda environment is `cocoa`. We also presume the user's terminal is in the folder where Cocoa was cloned.

:one: **Step 1 of 6**: activate the Conda Cocoa environment
    
        $ conda activate cocoa

:two: **Step 2 of 6**: go to the `projects` folder and clone the Cosmolike LSST-Y1 project:
    
        $(cocoa) cd ./cocoa/Cocoa/projects
        $(cocoa) git clone --depth 1 git@github.com:CosmoLike/cocoa_lsst_y1.git lsst_y1

The option `--depth 1` prevents git from downloading the entire project history. By convention, the Cosmolike Organization hosts a Cobaya-Cosmolike project named XXX at `CosmoLike/cocoa_XXX`. However, our scripts and YAML files assume the removal of the `cocoa_` prefix when cloning the repository.
 
:three: **Step 3 of 6**: go back to the Cocoa main folder, and activate the private Python environment
    
        $(cocoa) cd ../
        $(cocoa) source start_cocoa
 
:warning: (**warning**) :warning: Remember to run the start_cocoa script only after cloning the project repository. The script *start_cocoa* creates the necessary symbolic links and adds the *Cobaya-Cosmolike interface* of all projects to `LD_LIBRARY_PATH` and `PYTHONPATH` paths.

:four: **Step 4 of 6**: compile the project
 
        $(cocoa)(.local) source ./projects/lsst_y1/scripts/compile_lsst_y1

:five: **Step 5 of 6**: select the number of OpenMP cores
    
        $(cocoa)(.local) export OMP_PROC_BIND=close; export OMP_NUM_THREADS=4
        
:six:  **Step 6 of 6**: run a template YAML file

One model evaluation:

        $(cocoa)(.local) mpirun -n 1 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f
 
MCMC:

        $(cocoa)(.local) mpirun -n 4 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f

## Deleting Cosmolike projects <a name="running_cosmolike_projects"></a>

:warning: (**warning**) :warning: Never delete the `lsst_y1` folder from the project folder without running `stop_cocoa` first; otherwise, Cocoa will have ill-defined soft links at `Cocoa/cobaya/cobaya/likelihoods/`, `Cocoa/external_modules/code/` and `Cocoa/external_modules/data/`

## MCMC Convergence Criteria <a name="running_cosmolike_projects"></a>

  We are strict in our convergence criteria on `EXAMPLE_MCMC[0-9].YAML` MCMC examples.
  
    Rminus1_stop: 0.005
    # Gelman-Rubin R-1 on std deviations
    Rminus1_cl_stop: 0.15
    
For most applications, these settings are overkill (except when computing some tension and goodness of fit metrics). Please adjust these settings to your needs. 
