#include "pr.hpp"
#include <iostream>
#include <memory>
template <class T>
inline T ss_max(T x, T y)
{
    return (x < y) ? y : x;
}

template <class T>
inline T ss_min(T x, T y)
{
    return (y < x) ? y : x;
}
// \sigma = V * L * V^(-1)
glm::mat2 cov_of( float theta, float sx, float sy )
{
    float cosTheta = std::cosf(theta);
    float sinTheta = std::sinf(theta);
    float lambda0 = sx * sx;
    float lambda1 = sy * sy;
    float s11 = lambda0 * cosTheta * cosTheta + lambda1 * sinTheta * sinTheta;
    float s12 = (lambda0 - lambda1) * sinTheta * cosTheta;
    return glm::mat2(
        s11, s12,
        s12, lambda0 + lambda1 - s11);
}
// lambda0 is larger
void eignValues(float* lambda0, float* lambda1, float* determinant, const glm::mat2& mat)
{
    float mean = (mat[0][0] + mat[1][1]) * 0.5f;
    float det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
    float d = std::sqrtf(ss_max(mean * mean - det, 0.0f));
    *lambda0 = mean + d;
    *lambda1 = mean - d;
    *determinant = det;
}

void eigenVectors_of_symmetric(glm::vec2* eigen0, glm::vec2* eigen1, const glm::mat2& m, float lambda )
{
    float s11 = m[0][0];
    float s22 = m[1][1];
    float s12 = m[1][0];

    // to workaround lambda0 == lambda1
    float eps = 1e-15f;
    glm::vec2 e0 = glm::normalize(s11 < s22 ? glm::vec2(s12 + eps, lambda - s11) : glm::vec2(lambda - s22, s12 + eps));
    glm::vec2 e1 = { -e0.y, e0.x };
    *eigen0 = e0;
    *eigen1 = e1;
}


// https://www.shadertoy.com/view/XtGGzG

glm::vec3 magma_quintic(float x)
{
    x = glm::clamp(x, 0.0f, 1.0f);
    glm::vec4 x1 = glm::vec4(1.0f, x, x * x, x * x * x); // 1 x x2 x3
    glm::vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
    return glm::vec3(
        glm::dot(x1, glm::vec4(-0.023226960f, +1.087154378f, -0.109964741f, +6.333665763f)) + glm::dot(glm::vec2(x2), glm::vec2(-11.640596589f, +5.337625354f)),
        glm::dot(x1, glm::vec4(+0.010680993f, +0.176613780f, +1.638227448f, -6.743522237f)) + glm::dot(glm::vec2(x2), glm::vec2(+11.426396979f, -5.523236379f)),
        glm::dot(x1, glm::vec4(-0.008260782f, +2.244286052f, +3.005587601f, -24.279769818f)) + glm::dot(glm::vec2(x2), glm::vec2(+32.484310068f, -12.688259703f)));
}
glm::vec3 plasma_quintic(float x)
{
    x = glm::clamp(x, 0.0f, 1.0f);
    glm::vec4 x1 = glm::vec4(1.0f, x, x * x, x * x * x); // 1 x x2 x3
    glm::vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
    return glm::vec3(
        glm::dot(x1, glm::vec4(+0.063861086f, +1.992659096f, -1.023901152f, -0.490832805f)) + glm::dot(glm::vec2(x2), glm::vec2(+1.308442123f, -0.914547012f)),
        glm::dot(x1, glm::vec4(+0.049718590f, -0.791144343f, +2.892305078f, +0.811726816f)) + glm::dot(glm::vec2(x2), glm::vec2(-4.686502417f, +2.717794514f)),
        glm::dot(x1, glm::vec4(+0.513275779f, +1.580255060f, -5.164414457f, +4.559573646f)) + glm::dot(glm::vec2(x2), glm::vec2(-1.916810682f, +0.570638854f)));
}
float sign_of(float v)
{
    return v < 0.0f ? -1.0f : 1.0f;
}
// ax^2 + bx + c == 0
int solve_quadratic(float xs[2], float a, float b, float c)
{
    float det = b * b - 4.0f * a * c;
    if (det < 0.0f)
    {
        return 0;
    }

    float k = (-b - sign_of(b) * std::sqrtf(det)) / 2.0f;
    float x0 = k / a;
    float x1 = c / k;
    xs[0] = ss_min(x0, x1);
    xs[1] = ss_max(x0, x1);
    return 2;
}

float sqr(float x)
{
    return x * x;
}

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 0, 0, 10 };
    camera.lookat = { 0, 0, 0 };

    double e = GetElapsedTime();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();
        SetDepthTest(true);

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });

        static glm::vec3 mu = { 0, 0, 0 };

        glm::vec3 previous_mu = mu;
		ManipulatePosition(camera, &mu, 0.5f);
        mu.z = 0;
        glm::vec3 dmu = mu - previous_mu;

        static bool orthogonal = true;
        static glm::vec3 u_p = { 1.0f, 0, 0 };
        static glm::vec3 v_p = { 0, 1.0f, 0 };
        ManipulatePosition(camera, &u_p, 0.4f);
        u_p.z = 0;                       
        ManipulatePosition(camera, &v_p, 0.4f);
        v_p.z = 0;

        u_p += dmu;
        v_p += dmu;

        if( orthogonal )
        {
            glm::vec2 u = u_p - mu;
            float sy = glm::length(v_p - mu);
            glm::vec2 ortho_v = glm::normalize(glm::vec2(-u.y, u.x)) * sy;
            v_p = mu + glm::vec3(ortho_v, 0.0f);
        }


        glm::vec2 u = u_p - mu;
        glm::vec2 v = v_p - mu;

        //glm::vec2 un = glm::vec2(std::cosf(thetaR), std::sinf(thetaR));
        //glm::vec2 vn = glm::vec2(-std::sinf(thetaR), std::cosf(thetaR));

        DrawArrow(mu, mu + glm::vec3{ u.x, u.y, 0 }, 0.01f, { 255,0,0 });
        DrawArrow(mu, mu + glm::vec3{ v.x, v.y, 0 }, 0.01f, { 0,255,0 });
        DrawText(mu, "mu");
        DrawText(mu + glm::vec3{ u.x, u.y, 0 }, "u");
        DrawText(mu + glm::vec3{ v.x, v.y, 0 }, "v");

        //for (int k = 1; k <= 3; k++)
        //{
        //    PrimBegin( PrimitiveMode::LineStrip );
        //    for (int i = 0; i <= 64; i++)
        //    {
        //        float theta = ((float)i / 64) * glm::pi<float>() * 2.0f;
        //        glm::vec2 pe = glm::vec2(mu) + u * std::cosf(theta) * (float)k + v * std::sinf(theta) * (float)k;
        //        PrimVertex({ pe.x, pe.y, 0 }, { 255,255,255 });
        //    }
        //    PrimEnd();
        //}

        //DrawArrow(mu + glm::vec3(0,0,2), mu + glm::vec3(0, 0, 2) + glm::vec3{ UX,UY, 0}, 0.01f, {0,255,255 });
        //DrawText(mu + glm::vec3(0, 0, 2) + glm::vec3{ UX,UY, 0 }, "UXY");

        // vector formulation
        glm::mat2 inv_cov;
        {
            //float inv_uu = 1.0f / glm::dot(u, u);
            //float inv_vv = 1.0f / glm::dot(v, v);
            //float inv_uu2 = inv_uu * inv_uu;
            //float inv_vv2 = inv_vv * inv_vv;
            //float a = u.x * u.x * inv_uu2 + v.x * v.x * inv_vv2;
            //float b = u.x * u.y * inv_uu2 + v.x * v.y * inv_vv2;
            //float d = u.y * u.y * inv_uu2 + v.y * v.y * inv_vv2;

            //inv_cov = glm::mat2(
            //    a, b,
            //    b, d
            //);
            //auto R = glm::mat2(glm::normalize(u), glm::normalize(v));
            //inv_cov = R * glm::mat2(inv_uu, 0, 0, inv_vv) * glm::inverse(R);

            float a = u.x * u.x + v.x * v.x;
            float b = u.x * u.y + v.x * v.y;
            float d = u.y * u.y + v.y * v.y;
            inv_cov = glm::mat2(
                a, b,
                b, d
            );
        }

        float det_of_invcov;
        float lambda0_inv;
        float lambda1_inv;
        eignValues(&lambda0_inv, &lambda1_inv, &det_of_invcov, inv_cov);

        glm::vec2 e0;
        glm::vec2 e1;
        eigenVectors_of_symmetric(&e0, &e1, inv_cov, lambda0_inv);

        glm::vec3 e0_p = mu + glm::vec3(e0 / std::sqrt( lambda0_inv ), 0.0f);
        glm::vec3 e1_p = mu + glm::vec3(e1 / std::sqrt( lambda1_inv ), 0.0f);

        glm::vec3 depth = glm::vec3(0, 0, 0.1f);
        DrawArrow(mu + depth, e0_p + depth, 0.01f, { 255, 255, 0 });
        DrawArrow(mu + depth, e1_p + depth, 0.01f, { 255, 255, 0 });
        DrawText(e0_p + depth, "eigen0", 16, { 255, 0, 0 });
        DrawText(e1_p + depth, "eigen1", 16, { 255, 0, 0 });

        for (int k = 1; k <= 3; k++)
        {
            float sqLamda0 = std::sqrt(1.0f / lambda0_inv);
            float sqLamda1 = std::sqrt(1.0f / lambda1_inv);
            PrimBegin(PrimitiveMode::LineStrip);
            for (int i = 0; i <= 64; i++)
            {
                float theta = ((float)i / 64) * glm::pi<float>() * 2.0f;
                glm::vec2 pe = glm::vec2(mu) + e0 * sqLamda0 * std::cosf(theta) * (float)k + e1 * sqLamda1 * std::sinf(theta) * (float)k;
                PrimVertex({ pe.x, pe.y, 0 }, { 255,255,255 });
            }
            PrimEnd();
        }


        //printf("l %f %f\n", lambda0, lambda1);
        //printf("u v %f %f\n", glm::dot(u, u), glm::dot(v, v));

        //glm::mat2 cov = cov_of( thetaR, sx, sy );

        //auto rot2d = []( float rad ) {
      	 //   float cosTheta = std::cosf( rad );
      	 //   float sinTheta = std::sinf( rad );
      	 //   return glm::mat2( cosTheta, sinTheta, -sinTheta, cosTheta);
        //};

        //glm::mat2 R = rot2d( thetaR );
        //glm::mat2 cov2 = R * glm::mat2(
        //    glm::dot(u, u), 0.0f,
      	 //   0.0f, glm::dot(v, v)
        //) * glm::transpose(R);

        //glm::vec2 un = glm::vec2(std::cosf(thetaR), std::sinf(thetaR));
        //glm::vec2 vn = glm::vec2(-std::sinf(thetaR), std::cosf(thetaR));
        //glm::mat2 R2 = {
        //    un.x, un.y,
        //    vn.x, vn.y,
        //};

        // det = det(cov) = 1 / det(inv_cov)

        //float det;
        //float lambda0;
        //float lambda1;
        //eignValues(&lambda0, &lambda1, &det, cov);

        //glm::mat2 inv_cov =
        //    glm::mat2(
        //        cov[1][1], -cov[0][1],
        //        -cov[1][0], cov[0][0]) /
        //    det;

        for ( float y = - 5; y < 5 ; y += 0.05f )
        {
            PrimBegin(PrimitiveMode::LineStrip);
            for (float x = -5; x < 5; x += 0.05f)
            {
                glm::vec2 p = { x, y };
                glm::vec2 in_v = p - glm::vec2(mu.x, mu.y);

                float d2 = glm::dot(in_v, inv_cov * in_v);
                float alpha = glm::exp(-0.5f * d2);

                d2 = sqr(glm::dot(u, in_v)) + sqr(glm::dot(v, in_v));

                glm::u8vec3 color = glm::u8vec3( glm::clamp(plasma_quintic( alpha ) * 255.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(255.0f, 255.0f, 255.0f)) );
                PrimVertex(glm::vec3(x, y, alpha * 1.0f), color);
            }
            PrimEnd();

            // Minimum range of x
            for (int k = 1; k <= 3; k++)
            {
                float vy = y - mu.y;
                float a = inv_cov[0][0];
                float b = inv_cov[1][0];
                float d = inv_cov[1][1];
                float xs[2];
                if( solve_quadratic(xs, a, 2.0f * b * vy, d * vy * vy - k * k ) )
                {
                    int col = 250 -k * 30;
                    DrawLine(glm::vec3(mu.x + xs[0], y, 0), glm::vec3(mu.x + xs[1], y, 0), { col,col,col }, 3 - k);
                }
            }
        }

        // dp/dtheta
        //for (int i = 0; i <= 10; i++)
        //{
        //    float theta = ((float)i / 10) * glm::pi<float>() * 2.0f;
        //    glm::vec2 pe = glm::vec2(mu) + u * std::cosf(theta) + v * std::sinf(theta);
        //    glm::vec2 dp_dtheta = -u * std::sinf(theta) + v * std::cosf(theta);
        //    DrawArrow(glm::vec3(pe.x, pe.y, 0), glm::vec3(pe.x, pe.y, 0) + glm::vec3(dp_dtheta.x, dp_dtheta.y, 0), 0.1f, { 255, 255, 0 });
        //}

        // The exact bounding box from eigen vectors
        glm::vec2 hsize = {
            std::sqrt( u.x * u.x + v.x * v.x ),
            std::sqrt( u.y * u.y + v.y * v.y ),
        };
        //DrawCube(mu, glm::vec3(hsize.x, hsize.y, 0.0f) * 2.0f, { 255, 255, 255 });

        // The exact bounding box from inverse of covariance matrix
        for (int k = 1; k <= 3; k++)
        {
            float eps = 1e-15f;
            float hsize_invCovX = std::sqrt(inv_cov[1][1] / ( det_of_invcov + eps )) * (float)k;
            float hsize_invCovY = std::sqrt(inv_cov[0][0] / ( det_of_invcov + eps )) * (float)k;
            DrawCube(mu, glm::vec3(hsize_invCovX, hsize_invCovY, 0.0f) * 2.0f, { 255, 255, 255 });
        }
        
        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        //ImGui::SliderFloat("sx", &sx, 0.01f, 32);
        //ImGui::SliderFloat("sy", &sy, 0.01f, 32);
        //ImGui::SliderFloat("theta", &thetaR, 0, glm::pi<float>() * 2);
        // ImGui::SliderFloat("thetaR2", &thetaR2, 0, glm::pi<float>() * 2);
        ImGui::Checkbox( "orthogonal", &orthogonal );

        if (ImGui::Button("Make UV orthogonal"))
        {
            u = e0 * std::sqrtf(lambda0_inv);
            v = e1 * std::sqrtf(lambda1_inv);
            u_p = mu + glm::vec3(u, 0);
            v_p = mu + glm::vec3(v, 0);
        }

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
