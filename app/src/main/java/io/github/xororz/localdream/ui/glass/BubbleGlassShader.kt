package io.github.xororz.localdream.ui.glass

import android.graphics.RenderEffect
import android.graphics.RuntimeShader
import androidx.compose.ui.graphics.asComposeRenderEffect

/** Refracts the recorded cover itself. The moving boundary and optical normal share one field. */
internal class BubbleGlassShader {
    private val shader = RuntimeShader(SOURCE)

    fun effect(
        width: Float,
        height: Float,
        phase: Float,
        touchX: Float,
        touchY: Float,
    ) = run {
        shader.setFloatUniform("resolution", width.coerceAtLeast(1f), height.coerceAtLeast(1f))
        shader.setFloatUniform("phase", phase)
        shader.setFloatUniform("touch", touchX, touchY)
        RenderEffect.createRuntimeShaderEffect(shader, "content").asComposeRenderEffect()
    }

    private companion object {
        const val SOURCE = """
            uniform shader content;
            uniform float2 resolution;
            uniform float phase;
            uniform float2 touch;

            float field(float2 p) {
                float a = atan(p.y, p.x);
                float strength = min(length(touch), 1.0);
                float direction = atan(touch.y, touch.x + 0.0001);
                float r = 0.77 + 0.045*sin(3.0*a+phase) + 0.025*cos(2.0*a-phase*0.8);
                // Quadrupole stretches the contour along the gesture instead of merely shifting it.
                r += 0.095*strength*cos(2.0*(a-direction));
                r += 0.025*strength*sin(3.0*a-phase*1.3);
                return length(p)-r;
            }

            half4 main(float2 xy) {
                float unit = min(resolution.x, resolution.y)*0.5;
                float2 p = (xy-resolution*0.5)/unit;
                float d = field(p);
                float mask = 1.0-smoothstep(-1.2/unit, 0.8/unit, d);
                if (mask <= 0.0) return half4(0.0);
                float e = 0.003;
                float2 n = normalize(float2(field(p+float2(e,0))-field(p-float2(e,0)),
                                           field(p+float2(0,e))-field(p-float2(0,e)))+float2(0.0001));
                float edge = 1.0-smoothstep(0.0, 0.22, -d);
                // A curved meniscus bends source pixels most strongly near the moving edge.
                float bend = sin(edge*1.570796)*edge;
                float2 sampleAt = xy-n*bend*unit*0.17;
                sampleAt = clamp(sampleAt, float2(1), resolution-float2(1));
                half4 base = content.eval(sampleAt);
                float split = bend*0.65;
                base.r = content.eval(clamp(sampleAt+n*split, float2(1), resolution-float2(1))).r;
                base.b = content.eval(clamp(sampleAt-n*split, float2(1), resolution-float2(1))).b;
                float2 light = normalize(float2(-0.6,-0.8)+touch*0.35);
                float fresnel = pow(edge, 2.8);
                float highlight = pow(max(dot(n,light),0.0), 8.0)*fresnel*0.82;
                float shade = max(dot(n,-light),0.0)*fresnel*0.18;
                float rim = pow(edge, 12.0)*0.16;
                base.rgb = base.rgb*(1.0-shade)+half3(highlight+rim)*base.a;
                return base*half(mask);
            }
        """
    }
}
